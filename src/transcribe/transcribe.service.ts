import { Injectable, Logger } from '@nestjs/common';
import { AssemblyAI } from 'assemblyai';
import Anthropic from '@anthropic-ai/sdk';

// ─── Timing ──────────────────────────────────────────────────────────────────
const T_SILENCE_CLOSE         = 1200;   // 1200ms: con AAI EOT desactivado, el código controla cierres
                                         // 1200ms captura pausas naturales del doctor entre oraciones
                                         // sin fragmentar bloques. Antes era 800ms pero AAI cerraba antes.
const MIN_SPEAKER_CHANGE_CONF = 0.72;


// ─── Tipos ───────────────────────────────────────────────────────────────────
interface TurnBuffer {
  text:            string;
  lang:            'es' | 'en' | null;
  lastUpdateMs:    number;
  lastClosedMs:    number;
  lastEmittedText: string;
  lastEmittedLang: 'es' | 'en' | null;
  timer:           NodeJS.Timeout | null;
  lastSeenText:    string;   // Para detectar Turn estancado (texto repetido sin cambio)
  staleCount:      number;   // Cuántas veces consecutivas llegó el mismo texto
  forceClosedMs:   number;   // Timestamp del último ForceClose — bloquea ContinuationGuard

}

interface ConversationTurn { lang: 'es' | 'en'; text: string; }

interface SessionData {
  buffer:              TurnBuffer;
  conversationHistory: ConversationTurn[];
  chunkCount:          number;
  callback:            (data: string) => void;
}

@Injectable()
export class TranscribeService {
  private readonly logger = new Logger(TranscribeService.name);
  private assembly:   AssemblyAI | null = null;
  private anthropic:  Anthropic  | null = null;
  private sessionData = new Map<string, SessionData>();

  constructor() {
    const assemblyKey = process.env.ASSEMBLYAI_API_KEY;
    const claudeKey   = process.env.ANTHROPIC_API_KEY;
    if (assemblyKey) {
      this.assembly = new AssemblyAI({ apiKey: assemblyKey });
      this.logger.log('✅ AssemblyAI listo');
    } else {
      this.logger.warn('⚠️  ASSEMBLYAI_API_KEY no configurada');
    }
    if (claudeKey) {
      this.anthropic = new Anthropic({ apiKey: claudeKey });
      this.logger.log('✅ Claude Haiku listo');
    }
  }

  async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
    if (!this.assembly) return { text: '' };
    const t = await this.assembly.transcripts.transcribe({
      audio: file.buffer, language_code: 'es',
    });
    return { text: t.text || '' };
  }

  // ─── Buffer ────────────────────────────────────────────────────────────────

  private emptyBuf(): TurnBuffer {
    return { text: '', lang: null, lastUpdateMs: 0, lastClosedMs: 0, forceClosedMs: 0,
             lastEmittedText: '', lastEmittedLang: null, timer: null,
             lastSeenText: '', staleCount: 0 };
  }

  private clearTimer(buf: TurnBuffer) {
    if (buf.timer) { clearTimeout(buf.timer); buf.timer = null; }
  }

  private resetBuffer(buf: TurnBuffer) {
    buf.text = ''; buf.lang = null; buf.lastUpdateMs = 0; buf.timer = null;
    buf.lastSeenText = ''; buf.staleCount = 0;
    // lastClosedMs / lastEmittedText / lastEmittedLang se preservan para dedup
  }

  // ─── Idioma ────────────────────────────────────────────────────────────────

  private detectLang(text: string): 'es' | 'en' {
    const t = text.toLowerCase();
    const esScore = (t.match(
      /\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar)\b/g,
    ) || []).length;
    const enScore = (t.match(
      /\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes)\b/g,
    ) || []).length;
    return esScore > enScore ? 'es' : 'en';
  }

  // Retorna { lang, strongSignal } — strongSignal=true cuando hay evidencia clara del idioma
  private detectLangWithStrength(text: string): { lang: 'es' | 'en'; strong: boolean } {
    const t = text.toLowerCase();
    const esScore = (t.match(
      /\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar)\b/g,
    ) || []).length;
    const enScore = (t.match(
      /\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes)\b/g,
    ) || []).length;
    const lang  = esScore > enScore ? 'es' : 'en';
    const strong = Math.max(esScore, enScore) >= 2 || Math.abs(esScore - enScore) >= 2;
    return { lang, strong };
  }

  private resolveLang(
    text: string, aaiLang: string | undefined, aaiConf: number,
    bufLang: 'es' | 'en' | null, wordCount: number,
  ): 'es' | 'en' {
    // 1. AAI confiable → confiar siempre
    if (aaiLang && aaiConf > 0.55) return aaiLang.startsWith('es') ? 'es' : 'en';
    // 2. Léxico con señal fuerte → usar aunque sea texto corto
    const { lang: lexLang, strong } = this.detectLangWithStrength(text);
    if (strong) return lexLang;
    // 3. Texto muy corto sin evidencia → mantener idioma activo para no flipear
    if (wordCount <= 2 && bufLang) return bufLang;
    // 4. Fallback léxico
    return lexLang;
  }

  // ─── Texto ─────────────────────────────────────────────────────────────────

  private fixText(text: string, lang: 'es' | 'en'): string {
    let t = text.trim();
    t = t.replace(/\b(keprah?|kepra|quepra|kephra|kebri[ah]?|kebra)\b/gi, 'Keppra');
    if (lang === 'es') t = t.replace(/^(see|si)\s/i, 'Sí, ').replace(/\b2[\s,]?000\b/g, '2,000');
    if (lang === 'en') t = t.replace(/\b2[\s,]?000\b/g, '2,000');

    // Limpiar prefijo numérico suelto cuando el resto es EN puro.
    // Caso: "4 or after the dose increase." → "Before or after the dose increase."
    // AAI funde la respuesta del paciente ("4") con la pregunta del doctor.
    // Si el texto empieza con 1-2 palabras que son números/respuestas cortas
    // seguidas de palabras claramente EN, quitar el prefijo.
    const enStartWords = /^(or|before|after|the|was|were|is|are|have|had|do|does|did|when|where|what|how|why|which|that|this|it|in|of|for|with|a|an|and|but|not|no|any|all|one|two|three|four|some|your|their|our|my|its)/i;
    const shortPrefixMatch = t.match(/^(\d{1,3}\.?\s+)(\w.+)/);
    if (shortPrefixMatch && enStartWords.test(shortPrefixMatch[2])) {
      // El prefijo es un número y el resto parece oración EN
      t = shortPrefixMatch[2].charAt(0).toUpperCase() + shortPrefixMatch[2].slice(1);
    }

    const firstWord = t.split(/\s+/)[0]?.replace(/[.,!?¿¡]/g, '').toLowerCase() ?? '';
    const isCont = /^(pude|pudo|puede|me|te|se|lo|la|le|los|las|y|e|o|pero|que|porque|aunque|cuando|and|or|but|so|because|since|though|however)$/.test(firstWord);
    if (!isCont && t.length > 0) t = t.charAt(0).toUpperCase() + t.slice(1);
    return t;
  }

  private norm(s: string): string {
    return s.replace(/[.,;:!?¿¡]/g, '').toLowerCase().replace(/\s+/g, ' ').trim();
  }

  private isBackchannel(text: string): boolean {
    const t = text.trim().replace(/[.!?¿¡,]/g, '').toLowerCase();
    if (/^\d{1,3}$/.test(t)) return true;
    return /^(sí|si|no|okay|ok|claro|bueno|bien|ajá|aja|mhm|yes|yeah|nope|cuatro|four|tres|three|dos|two|uno|one)$/.test(t);
  }

  // ─── Emit ──────────────────────────────────────────────────────────────────

  private emit(session: SessionData, payload: object) {
    session.callback(JSON.stringify(payload));
  }

  private emitPartial(session: SessionData, sessionId: string) {
    const buf = session.buffer;
    if (!buf.text || !buf.lang) return;
    // No emitir partials de 1 sola palabra — AAI a veces emite texto basura
    // de 1 palabra durante la clasificación de idioma (ej: "See", "those", "me")
    // que luego descarta. Esperar al menos 2 palabras antes de mostrar al usuario.
    const words = buf.text.trim().split(/\s+/).filter(Boolean).length;
    if (words < 2) return;
    this.emit(session, { text: buf.text, language: buf.lang, isNewTurn: false, sessionId });
  }

  // ─── Cierre de turno ───────────────────────────────────────────────────────

  private async closeTurn(sessionId: string, reason: string): Promise<void> {
    const session = this.sessionData.get(sessionId);
    if (!session) return;
    const buf = session.buffer;
    if (!buf.text) return;

    this.clearTimer(buf);
    const lang      = buf.lang ?? this.detectLang(buf.text);
    const finalText = this.fixText(buf.text, lang);
    if (!finalText) { this.resetBuffer(buf); return; }

    // ── Filtro de eco: 1 sola palabra que coincide con final de bloques recientes ──
    // "Increase." aparece como eco del "medication increase." aunque no sea el bloque inmediato anterior.
    // Buscar en los últimos 5 bloques del historial de conversación.
    const wordCount = finalText.trim().split(/\s+/).length;
    if (wordCount === 1) {
      const w = this.norm(finalText);
      // Verificar bloque inmediato anterior
      const prev = this.norm(buf.lastEmittedText ?? '');
      if (prev.endsWith(w)) {
        this.logger.log(`🔇 Eco descartado [${lang}] [${sessionId}]: "${finalText}"`);
        this.resetBuffer(buf);
        return;
      }
      // Verificar los últimos 5 bloques del historial
      const recentHistory = session.conversationHistory.slice(-5);
      for (const h of recentHistory) {
        if (this.norm(h.text).endsWith(w)) {
          this.logger.log(`🔇 Eco descartado (hist) [${lang}] [${sessionId}]: "${finalText}"`);
          this.resetBuffer(buf);
          return;
        }
      }
    }

    // ── Dedup: no emitir si es idéntico al bloque anterior ─────────────────────
    // EXCEPCIÓN: backchannels cortos (≤2 palabras) siempre se emiten aunque
    // sean iguales — el paciente puede decir "No." / "Sí." varias veces seguidas.
    const isShortBackchannel = wordCount <= 2;
    if (!isShortBackchannel && this.norm(finalText) === this.norm(buf.lastEmittedText)) {
      this.logger.log(`⏭ Dedup skip [${lang}] [${sessionId}]`);
      this.resetBuffer(buf);
      return;
    }

    this.logger.log(`✅ CLOSE [${lang}] [${sessionId}] (${reason}): "${finalText.substring(0, 80)}"`);
    buf.lastEmittedText = finalText;
    buf.lastEmittedLang = lang;
    buf.lastClosedMs    = Date.now();

    // Emitir el bloque — Claude corre en background
    this.emit(session, { text: finalText, language: lang, isNewTurn: true, isForcedClose: false, sessionId });
    session.conversationHistory.push({ lang, text: finalText });
    if (session.conversationHistory.length > 20) session.conversationHistory.shift();

    this.resetBuffer(buf);
    this.claudePipeline(finalText, lang, session, sessionId);
  }

  // ─── Transcripción en tiempo real ─────────────────────────────────────────

  async startRealTimeTranscription(
    sessionId: string,
    callback: (data: string) => void,
  ): Promise<{ send: (chunk: ArrayBuffer) => void; close: () => void }> {
    const apiKey = process.env.ASSEMBLYAI_API_KEY;
    if (!apiKey) throw new Error('ASSEMBLYAI_API_KEY no configurada');

    const session: SessionData = {
      buffer: this.emptyBuf(), conversationHistory: [], chunkCount: 0, callback,
    };
    this.sessionData.set(sessionId, session);
    this.logger.log(`🎤 AssemblyAI v3 iniciando [${sessionId}]`);

    const params = new URLSearchParams({
      sample_rate: '16000', format_turns: 'true',
      speech_model: 'universal-streaming-multilingual', language_detection: 'true',
      end_of_turn_confidence_threshold: '0.5',
      max_turn_silence: '800',
    });

    const KEYTERMS = ['Keppra','convulsión','convulsiones','epilepsia',
      'seizure','seizures','levetiracetam','medicamento','medicamentos',
      'valproato','carbamazepina','lamotrigina','cerebro','dosis'];

    const WebSocket = require('ws');
    const ws = new WebSocket(
      `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`,
      { headers: { Authorization: apiKey } },
    );

    ws.on('open',  () => this.logger.log(`✅ AssemblyAI v3 abierto [${sessionId}]`));
    ws.on('error', (err: Error) => this.logger.error(`❌ AssemblyAI v3 error [${sessionId}]: ${err.message}`));

    ws.on('close', (code: number) => {
      this.logger.log(`🔒 AssemblyAI v3 cerrado [${sessionId}] (${code})`);
      const s = this.sessionData.get(sessionId);
      if (s?.buffer.text) this.closeTurn(sessionId, 'streamClose');
      this.sessionData.delete(sessionId);
    });

    ws.on('message', (raw: any) => {
      const s = this.sessionData.get(sessionId);
      if (!s) return;
      let msg: any;
      try { msg = JSON.parse(raw.toString()); } catch { return; }

      const buf = s.buffer;
      const now = Date.now();

      if (msg.type === 'Begin') {
        this.logger.log(`🔗 Sesión AssemblyAI v3 [${sessionId}] sid=${msg.id}`);
        ws.send(JSON.stringify({ type: 'UpdateConfiguration', keyterms: KEYTERMS }));
        this.logger.log(`📚 Keyterms enviados [${sessionId}]`);
        return;
      }

      if (msg.type === 'Turn') {
        const text: string     = (msg.transcript || '').trim();
        const aaiLang: string  = msg.language_code;
        const aaiConf: number  = msg.language_confidence ?? 0;
        const isFinal: boolean = msg.turn_is_formatted === true;
        const wordCount        = text.split(/\s+/).filter(Boolean).length;

        this.logger.log(
          `🔬 RAW [${sessionId}] fmt=${isFinal} lang=${aaiLang} conf=${aaiConf.toFixed(2)} text="${text.substring(0,60)}"`,
        );
        if (!text) return;

        // ── Filtro de ruido: rechazar si AAI detectó idioma != es/en con conf baja ─
        // Ruido ambiental produce transcripciones en fr/it/pt con conf < 0.35
        // y texto corto (1-2 palabras). Ejemplo: lang=fr conf=0.59 text="Conditions."
        // EXCEPCIÓN: palabras universales como "No/Si/Sí/Yes/Ok" no se descartan
        // porque son válidas en múltiples idiomas y son respuestas médicas importantes.
        const isNonTargetLang = aaiLang && aaiLang !== 'en' && aaiLang !== 'es' && aaiLang !== 'undefined';
        const isUniversalWord = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(text.trim());
        if (isNonTargetLang && aaiConf < 0.65 && wordCount <= 2 && !isUniversalWord) {
          this.logger.log(`🚫 Ruido descartado [${aaiLang}=${aaiConf.toFixed(2)}] "${text}" [${sessionId}]`);
          return;
        }

        // ── Guard de continuación post-close ─────────────────────────────────
        // AAI v3 sigue emitiendo Turns del mismo utterance después de que el
        // silence timer ya cerró el bloque. Si el buffer está vacío, se cerró
        // hace < 1200ms, y el texto nuevo empieza con los primeros ~20 chars
        // del bloque anterior → es continuación. Reabrimos el buffer en silencio.
        //
        // EXCEPCIÓN CRÍTICA: si el cierre fue un ForceClose por mezcla EN+ES,
        // NO reabrir — AAI sigue enviando el mismo Turn fusionado y si reabrimos
        // volvemos a acumular texto mezclado. Bloqueamos por 2000ms post-ForceClose.
        const msSinceClose = now - buf.lastClosedMs;
        const msSinceForceClose = now - buf.forceClosedMs;
        const forceCloseBlackout = msSinceForceClose < 2000;
        if (!buf.text && msSinceClose < 1200 && buf.lastEmittedText && !forceCloseBlackout) {
          const normalize = (s: string) => s
            .replace(/Keppra/gi, 'kepra').replace(/[,\.!?¿¡]/g, '')
            .replace(/\s+/g, ' ').trim().toLowerCase();
          const prevNorm = normalize(buf.lastEmittedText);
          const curNorm  = normalize(text);
          const prefix   = prevNorm.substring(0, Math.min(prevNorm.length, 20));
          if (prefix.length >= 4 && curNorm.startsWith(prefix)) {
            this.logger.log(`🔁 ContinuationGuard reopen [${sessionId}] +${msSinceClose}ms`);
            buf.text = text;
            buf.lang = buf.lastEmittedLang;
            this.clearTimer(buf);
            buf.timer = setTimeout(() => {
              buf.timer = null;
              this.logger.log(`⏱ Silence close [${sessionId}]`);
              this.closeTurn(sessionId, 'silence');
            }, T_SILENCE_CLOSE);
            return;
          }
        }

        // Detectar idioma con señal de fuerza léxica
        const { lang: lexLang, strong: lexStrong } = this.detectLangWithStrength(text);
        const detectedLang = this.resolveLang(text, aaiLang, aaiConf, buf.lang, wordCount);
        if (aaiLang) this.logger.log(`🌐 ASR lang=${aaiLang} conf=${aaiConf.toFixed(3)} words=${wordCount} → ${detectedLang} (lex=${lexLang} strong=${lexStrong})`);

        // ── Asignar idioma al buffer ────────────────────────────────────
        const bufEmpty = !buf.lang || !buf.text;
        if (bufEmpty) {
          if (lexStrong && buf.lastEmittedLang && buf.lastEmittedLang !== lexLang) {
            // Caso B: léxico fuerte señala idioma diferente al turno previo.
            buf.lang = lexLang;
            this.logger.log(`🌍 LangFromLex [${buf.lastEmittedLang}→${lexLang}] post-close [${sessionId}]`);
          } else if (!lexStrong && this.isBackchannel(text) && buf.lastEmittedLang) {
            // Caso C: backchannel ambiguo (No/Sí/Ok/4...) sin señal léxica fuerte.
            // Asumir idioma CONTRARIO al turno anterior (doctor EN → paciente ES y viceversa).
            // EXCEPCIÓN: Si el backchannel es "Si/Sí/No" y el turno anterior fue ES,
            // NO invertir — es más probable que el paciente continúe en ES que el doctor
            // diga "see?" o "no?" en ese momento. En ese caso mantener ES.
            const isSpanishBackchannel = /^(sí|si|no)\.?,?$/i.test(text.trim());
            if (isSpanishBackchannel && buf.lastEmittedLang === 'es') {
              buf.lang = 'es';
              this.logger.log(`🔄 BackchanelKeep [es] text="${text}" [${sessionId}]`);
            } else {
              const opposite = buf.lastEmittedLang === 'en' ? 'es' : 'en';
              buf.lang = opposite;
              this.logger.log(`🔄 BackchanelFlip [${buf.lastEmittedLang}→${opposite}] text="${text}" [${sessionId}]`);
            }
          } else {
            buf.lang = detectedLang;
          }
        } else if (aaiLang && aaiConf > 0.80) {
          buf.lang = detectedLang;
        }

        // ── Speaker change ──────────────────────────────────────────────
        // REGLA CRÍTICA: solo disparar si hay un silencio real entre hablantes.
        // Si el texto del buffer sigue creciendo activamente (lastUpdateMs reciente),
        // es el MISMO hablante — no importa si AAI cambia su estimación de idioma
        // a mitad de un utterance. "Keppra Y lo dejé..." empieza como EN y luego
        // AAI lo reclasifica como ES — sin silencio entre medio, no es speaker change.
        //
        // GUARD GEOMÉTRICO: si el texto nuevo EMPIEZA con el texto del buffer,
        // es el mismo hablante creciendo — imposible que sea speaker change.
        // "Keppra Y" → "Keppra Y lo" → startsWith → mismo turno, sin cambio.
        const isGrowingTurn = buf.text && text.startsWith(buf.text.trimEnd());

        // Condiciones para speaker change (TODAS deben cumplirse):
        // 1. El texto no está creciendo (guard geométrico)
        // 2. Silencio real: > 400ms desde el último Turn event
        // 3. Idioma detectado con confianza alta
        const silenceGap     = now - buf.lastUpdateMs > 400;
        const confOk         = aaiConf >= MIN_SPEAKER_CHANGE_CONF && wordCount >= 2;
        const veryConf       = aaiConf >= 0.80;
        const lexConfChange  = lexStrong && buf.lang && buf.lang !== lexLang && buf.text;
        const bufLangChanged = buf.lang && buf.lang !== detectedLang && buf.text;

        if (!isGrowingTurn && silenceGap && (
          (bufLangChanged && (confOk || veryConf)) ||
          (lexConfChange && wordCount >= 3)
        )) {
          this.logger.log(`🔀 SpeakerChange [${buf.lang}→${detectedLang}] gap=${now - buf.lastUpdateMs}ms [${sessionId}]`);
          this.closeTurn(sessionId, 'speakerChange');
          buf.lang = detectedLang;
        }

        buf.lastUpdateMs = now;

        // ── Acumular texto en buffer + emitir partial en vivo ──────────
        buf.text = text;
        this.emitPartial(s, sessionId);
        this.logger.log(`📝 ${isFinal ? 'FINAL' : 'Partial'} [${buf.lang}] [${sessionId}]: "${text.substring(0,80)}"`);

        // ── ForceClose por mezcla de idiomas en mismo Turn ──────────────
        // Cuando AAI fusiona doctor+paciente en un mismo Turn, el texto
        // acumulado contiene frases EN seguidas de frases ES (o viceversa).
        // Al detectar mezcla con ≥8 palabras, cerramos INMEDIATAMENTE y
        // retornamos para que el silence timer normal no sobreescriba.
        if (wordCount >= 8 && buf.text) {
          const words = text.trim().split(/\s+/);
          const esOnlyWords = /^(que|los|las|del|una|con|para|pero|desde|hace|porque|también|cuando|como|esto|eso|fue|han|tengo|tuve|tenía|convulsiones|días|mes|año|años|siempre|nunca|alguna|dejé|pagar|cobraba|incrementaron|tomarla|todos)$/i;
          const enOnlyWords = /^(the|and|you|have|had|are|taking|medications|seizures|since|before|after|dose|increase|missed|those|pills|times|every|medical|conditions|family|history|examine|when|was|your|last|seizure|not)$/i;
          const lastThird = words.slice(Math.floor(words.length * 0.6));
          const firstHalf = words.slice(0, Math.floor(words.length * 0.5));
          const firstHasEN = firstHalf.some(w => enOnlyWords.test(w));
          const firstHasES = firstHalf.some(w => esOnlyWords.test(w));
          const lastHasEN  = lastThird.some(w => enOnlyWords.test(w));
          const lastHasES  = lastThird.some(w => esOnlyWords.test(w));
          const mixDetected = (firstHasEN && lastHasES) || (firstHasES && lastHasEN);
          if (mixDetected) {
            this.logger.log(`🔀 ForceClose por mezcla EN+ES [${sessionId}] "${text.substring(0,60)}"`);
            this.clearTimer(buf);
            buf.forceClosedMs = now; // bloquear ContinuationGuard para este Turn de AAI
            this.closeTurn(sessionId, 'silence'); // cierre síncrono inmediato
            return; // no continuar al silence timer — ya cerramos
          }
        }
        // ── Silence timer: detectar Turn estancado ──────────────────────
        // Con end_of_turn_confidence_threshold=1.0, AAI nunca cierra su Turn
        // propio. Cuando el speaker hace pausa, AAI sigue enviando el MISMO
        // texto repetidamente (el buffer del Turn está "congelado"). En ese
        // caso NO resetear el timer — dejar que expire para cerrar el bloque.
        // Cuando el texto SÍ crece (nueva speech), resetar normalmente.
        const textGrew = text !== buf.lastSeenText;
        buf.lastSeenText = text;
        if (textGrew) {
          buf.staleCount = 0;
          // Texto nuevo → resetear timer
          this.clearTimer(buf);
          buf.timer = setTimeout(() => {
            buf.timer = null;
            this.logger.log(`⏱ Silence close [${sessionId}]`);
            this.closeTurn(sessionId, 'silence');
          }, T_SILENCE_CLOSE);
        } else {
          buf.staleCount++;
          // Texto estancado (mismo que antes) → NO resetear timer, dejar que expire
          // Loguear solo ocasionalmente para no saturar
          if (buf.staleCount === 3) {
            this.logger.log(`🧊 Turn estancado [${sessionId}] stale=${buf.staleCount} — timer no reseteado`);
          }
          // Si no hay timer activo (fue limpiado), crear uno nuevo de todas formas
          if (!buf.timer) {
            buf.timer = setTimeout(() => {
              buf.timer = null;
              this.logger.log(`⏱ Silence close [${sessionId}]`);
              this.closeTurn(sessionId, 'silence');
            }, T_SILENCE_CLOSE);
          }
        }

      } else if (msg.type === 'Termination') {
        this.logger.log(`🏁 Terminado [${sessionId}] audio=${msg.audio_duration_seconds}s`);
      }
    });

    const send = (chunk: ArrayBuffer) => {
      const s = this.sessionData.get(sessionId);
      if (!s) return;
      s.chunkCount++;
      if (s.chunkCount % 40 === 0) this.logger.log(`📤 [${sessionId}] Chunk #${s.chunkCount}`);
      if (ws.readyState === WebSocket.OPEN) ws.send(chunk);
    };

    const close = async () => {
      this.logger.log(`⏳ Cerrando AssemblyAI v3 [${sessionId}]`);
      const s = this.sessionData.get(sessionId);
      if (s?.buffer.text) await this.closeTurn(sessionId, 'userStop');
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'Terminate' }));
        // Esperar más tiempo para que AAI procese el audio en buffer antes de cerrar.
        // Con 800ms se perdían las últimas frases del doctor. Con 2500ms damos tiempo
        // suficiente para que AAI emita los Turns pendientes y los procesemos.
        await new Promise(r => setTimeout(r, 2500));
      }
      ws.close();
      this.logger.log(`🛑 AssemblyAI v3 cerrado [${sessionId}]`);
    };

    return { send, close };
  }

  // ─── Claude (background, no bloquea display) ──────────────────────────────

  private async claudePipeline(
    text: string, lang: 'es' | 'en',
    session: SessionData, sessionId: string,
  ) {
    const history = [...session.conversationHistory];
    const { result, correctedLang } = await this.correctWithClaude(text, lang, history);
    if (result !== text || correctedLang !== lang) {
      this.logger.log(`✨ CLAUDE [${lang}→${correctedLang}]: "${result.substring(0,80)}"`);
      const idx = session.conversationHistory.findLastIndex(t => t.text === text);
      if (idx >= 0) {
        session.conversationHistory[idx].text = result;
        session.conversationHistory[idx].lang = correctedLang;
      }
      // Si Claude corrigió el idioma, actualizar lastEmittedLang para que
      // el ContinuationGuard y BackchanelFlip usen el idioma correcto
      if (correctedLang !== lang && session.buffer.lastEmittedLang === lang) {
        session.buffer.lastEmittedLang = correctedLang;
      }
      this.emit(session, { text: result, language: correctedLang, isCorrection: true, originalText: text, sessionId });
    }
  }

  private async correctWithClaude(
    text: string, lang: 'es' | 'en', history: ConversationTurn[],
  ): Promise<{ result: string; correctedLang: 'es' | 'en' }> {
    if (!this.anthropic || text.length < 5) return { result: text, correctedLang: lang };
    const ctx = history.slice(0, -1).slice(-5)
      .map(t => `[${t.lang === 'en' ? 'Doctor' : 'Patient'}]: ${t.text}`)
      .join('\n');

    const prompt = `You are an ASR post-processor for a bilingual medical interpreter. Doctor speaks English, Patient speaks Spanish.
${ctx ? `Conversation so far:\n${ctx}\n` : ''}
ASR transcription to fix: "${text}"
Detected language: ${lang === 'es' ? 'Spanish (patient)' : 'English (doctor)'}

RULES — apply ONLY these corrections:
1. "kepra/keprah/kephra/quepra/kebra" → "Keppra"
2. Spanish "see " or "si " at utterance start → "Sí, "
3. "2000" in dosage context → "2,000"
4. Clear phonetic errors: "Wer you" → "Were you", "hav you" → "have you"
5. Fix obvious punctuation only
6. DO NOT add words, DO NOT complete sentences, DO NOT translate
7. If nothing to fix, return text EXACTLY as-is
8. CRITICAL — Wrong language detection: If the detected language is English but the text looks like garbled Spanish (e.g. "See those mean" could be "Si dos mil", "See" could be "Sí"), AND the conversation context shows the patient was just speaking Spanish about dosages, correct it to the most likely Spanish. Example: after patient says dosage info in Spanish, "See those mean." → "Sí, dos mil."

Output ONLY the corrected text — no explanations, no quotes.`;

    try {
      const r = await this.anthropic.messages.create({
        model: 'claude-haiku-4-5-20251001', max_tokens: 300,
        messages: [{ role: 'user', content: prompt }],
      });
      const result = (r.content[0] as any).text?.trim() || text;
      if (this.norm(result) === this.norm(text)) return { result: text, correctedLang: lang };
      if (result.length > text.length * 1.4 + 20) return { result: text, correctedLang: lang };
      // Detectar si Claude corrigió el idioma (ej: "See those mean" → "Sí, dos mil")
      const detectedResultLang = this.detectLang(result);
      const correctedLang: 'es' | 'en' = detectedResultLang ?? lang;
      return { result, correctedLang };
    } catch (e: any) {
      this.logger.error(`❌ Claude correct: ${e.message}`);
      return { result: text, correctedLang: lang };
    }
  }
}
// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';
// import Anthropic from '@anthropic-ai/sdk';

// // ─── Timing ──────────────────────────────────────────────────────────────────
// const T_SILENCE_CLOSE = 1200; // 1200ms: con AAI EOT desactivado, el código controla cierres
// // 1200ms captura pausas naturales del doctor entre oraciones
// // sin fragmentar bloques. Antes era 800ms pero AAI cerraba antes.
// const MIN_SPEAKER_CHANGE_CONF = 0.72;

// // ─── Tipos ───────────────────────────────────────────────────────────────────
// interface TurnBuffer {
//   text: string;
//   lang: 'es' | 'en' | null;
//   lastUpdateMs: number;
//   lastClosedMs: number;
//   lastEmittedText: string;
//   lastEmittedLang: 'es' | 'en' | null;
//   timer: NodeJS.Timeout | null;
//   lastSeenText: string; // Para detectar Turn estancado (texto repetido sin cambio)
//   staleCount: number; // Cuántas veces consecutivas llegó el mismo texto
//   forceClosedMs: number; // Timestamp del último ForceClose — bloquea ContinuationGuard
// }

// interface ConversationTurn {
//   lang: 'es' | 'en';
//   text: string;
// }

// interface SessionData {
//   buffer: TurnBuffer;
//   conversationHistory: ConversationTurn[];
//   chunkCount: number;
//   callback: (data: string) => void;
// }

// @Injectable()
// export class TranscribeService {
//   private readonly logger = new Logger(TranscribeService.name);
//   private assembly: AssemblyAI | null = null;
//   private anthropic: Anthropic | null = null;
//   private sessionData = new Map<string, SessionData>();

//   constructor() {
//     const assemblyKey = process.env.ASSEMBLYAI_API_KEY;
//     const claudeKey = process.env.ANTHROPIC_API_KEY;
//     if (assemblyKey) {
//       this.assembly = new AssemblyAI({ apiKey: assemblyKey });
//       this.logger.log('✅ AssemblyAI listo');
//     } else {
//       this.logger.warn('⚠️  ASSEMBLYAI_API_KEY no configurada');
//     }
//     if (claudeKey) {
//       this.anthropic = new Anthropic({ apiKey: claudeKey });
//       this.logger.log('✅ Claude Haiku listo');
//     }
//   }

//   async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
//     if (!this.assembly) return { text: '' };
//     const t = await this.assembly.transcripts.transcribe({
//       audio: file.buffer,
//       language_code: 'es',
//     });
//     return { text: t.text || '' };
//   }

//   // ─── Buffer ────────────────────────────────────────────────────────────────

//   private emptyBuf(): TurnBuffer {
//     return {
//       text: '',
//       lang: null,
//       lastUpdateMs: 0,
//       lastClosedMs: 0,
//       forceClosedMs: 0,
//       lastEmittedText: '',
//       lastEmittedLang: null,
//       timer: null,
//       lastSeenText: '',
//       staleCount: 0,
//     };
//   }

//   private clearTimer(buf: TurnBuffer) {
//     if (buf.timer) {
//       clearTimeout(buf.timer);
//       buf.timer = null;
//     }
//   }

//   private resetBuffer(buf: TurnBuffer) {
//     buf.text = '';
//     buf.lang = null;
//     buf.lastUpdateMs = 0;
//     buf.timer = null;
//     buf.lastSeenText = '';
//     buf.staleCount = 0;
//     // lastClosedMs / lastEmittedText / lastEmittedLang se preservan para dedup
//   }

//   // ─── Idioma ────────────────────────────────────────────────────────────────

//   private detectLang(text: string): 'es' | 'en' {
//     const t = text.toLowerCase();
//     const esScore = (
//       t.match(
//         /\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar)\b/g,
//       ) || []
//     ).length;
//     const enScore = (
//       t.match(
//         /\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes)\b/g,
//       ) || []
//     ).length;
//     return esScore > enScore ? 'es' : 'en';
//   }

//   // Retorna { lang, strongSignal } — strongSignal=true cuando hay evidencia clara del idioma
//   private detectLangWithStrength(text: string): {
//     lang: 'es' | 'en';
//     strong: boolean;
//   } {
//     const t = text.toLowerCase();
//     const esScore = (
//       t.match(
//         /\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar)\b/g,
//       ) || []
//     ).length;
//     const enScore = (
//       t.match(
//         /\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes)\b/g,
//       ) || []
//     ).length;
//     const lang = esScore > enScore ? 'es' : 'en';
//     const strong =
//       Math.max(esScore, enScore) >= 2 || Math.abs(esScore - enScore) >= 2;
//     return { lang, strong };
//   }

//   private resolveLang(
//     text: string,
//     aaiLang: string | undefined,
//     aaiConf: number,
//     bufLang: 'es' | 'en' | null,
//     wordCount: number,
//   ): 'es' | 'en' {
//     // 1. AAI confiable → confiar siempre
//     if (aaiLang && aaiConf > 0.55)
//       return aaiLang.startsWith('es') ? 'es' : 'en';
//     // 2. Léxico con señal fuerte → usar aunque sea texto corto
//     const { lang: lexLang, strong } = this.detectLangWithStrength(text);
//     if (strong) return lexLang;
//     // 3. Texto muy corto sin evidencia → mantener idioma activo para no flipear
//     if (wordCount <= 2 && bufLang) return bufLang;
//     // 4. Fallback léxico
//     return lexLang;
//   }

//   // ─── Texto ─────────────────────────────────────────────────────────────────

//   private fixText(text: string, lang: 'es' | 'en'): string {
//     let t = text.trim();
//     t = t.replace(
//       /\b(keprah?|kepra|quepra|kephra|kebri[ah]?|kebra)\b/gi,
//       'Keppra',
//     );
//     if (lang === 'es')
//       t = t.replace(/^(see|si)\s/i, 'Sí, ').replace(/\b2[\s,]?000\b/g, '2,000');
//     if (lang === 'en') t = t.replace(/\b2[\s,]?000\b/g, '2,000');

//     // Limpiar prefijo numérico suelto cuando el resto es EN puro.
//     // Caso: "4 or after the dose increase." → "Before or after the dose increase."
//     // AAI funde la respuesta del paciente ("4") con la pregunta del doctor.
//     // Si el texto empieza con 1-2 palabras que son números/respuestas cortas
//     // seguidas de palabras claramente EN, quitar el prefijo.
//     const enStartWords =
//       /^(or|before|after|the|was|were|is|are|have|had|do|does|did|when|where|what|how|why|which|that|this|it|in|of|for|with|a|an|and|but|not|no|any|all|one|two|three|four|some|your|their|our|my|its)/i;
//     const shortPrefixMatch = t.match(/^(\d{1,3}\.?\s+)(\w.+)/);
//     if (shortPrefixMatch && enStartWords.test(shortPrefixMatch[2])) {
//       // El prefijo es un número y el resto parece oración EN
//       t =
//         shortPrefixMatch[2].charAt(0).toUpperCase() +
//         shortPrefixMatch[2].slice(1);
//     }

//     const firstWord =
//       t
//         .split(/\s+/)[0]
//         ?.replace(/[.,!?¿¡]/g, '')
//         .toLowerCase() ?? '';
//     const isCont =
//       /^(pude|pudo|puede|me|te|se|lo|la|le|los|las|y|e|o|pero|que|porque|aunque|cuando|and|or|but|so|because|since|though|however)$/.test(
//         firstWord,
//       );
//     if (!isCont && t.length > 0) t = t.charAt(0).toUpperCase() + t.slice(1);
//     return t;
//   }

//   private norm(s: string): string {
//     return s
//       .replace(/[.,;:!?¿¡]/g, '')
//       .toLowerCase()
//       .replace(/\s+/g, ' ')
//       .trim();
//   }

//   private isBackchannel(text: string): boolean {
//     const t = text
//       .trim()
//       .replace(/[.!?¿¡,]/g, '')
//       .toLowerCase();
//     if (/^\d{1,3}$/.test(t)) return true;
//     return /^(sí|si|no|okay|ok|claro|bueno|bien|ajá|aja|mhm|yes|yeah|nope|cuatro|four|tres|three|dos|two|uno|one)$/.test(
//       t,
//     );
//   }

//   // ─── Emit ──────────────────────────────────────────────────────────────────

//   private emit(session: SessionData, payload: object) {
//     session.callback(JSON.stringify(payload));
//   }

//   private emitPartial(session: SessionData, sessionId: string) {
//     const buf = session.buffer;
//     if (!buf.text || !buf.lang) return;
//     // No emitir partials de 1 sola palabra — AAI a veces emite texto basura
//     // de 1 palabra durante la clasificación de idioma (ej: "See", "those", "me")
//     // que luego descarta. Esperar al menos 2 palabras antes de mostrar al usuario.
//     const words = buf.text.trim().split(/\s+/).filter(Boolean).length;
//     if (words < 2) return;
//     this.emit(session, {
//       text: buf.text,
//       language: buf.lang,
//       isNewTurn: false,
//       sessionId,
//     });
//   }

//   // ─── Cierre de turno ───────────────────────────────────────────────────────

//   private async closeTurn(sessionId: string, reason: string): Promise<void> {
//     const session = this.sessionData.get(sessionId);
//     if (!session) return;
//     const buf = session.buffer;
//     if (!buf.text) return;

//     this.clearTimer(buf);
//     const lang = buf.lang ?? this.detectLang(buf.text);
//     const finalText = this.fixText(buf.text, lang);
//     if (!finalText) {
//       this.resetBuffer(buf);
//       return;
//     }

//     if (this.norm(finalText) === this.norm(buf.lastEmittedText)) {
//       this.logger.log(`⏭ Dedup skip [${lang}] [${sessionId}]`);
//       this.resetBuffer(buf);
//       return;
//     }

//     this.logger.log(
//       `✅ CLOSE [${lang}] [${sessionId}] (${reason}): "${finalText.substring(0, 80)}"`,
//     );
//     buf.lastEmittedText = finalText;
//     buf.lastEmittedLang = lang;
//     buf.lastClosedMs = Date.now();

//     // Emitir el bloque — Claude corre en background
//     this.emit(session, {
//       text: finalText,
//       language: lang,
//       isNewTurn: true,
//       isForcedClose: false,
//       sessionId,
//     });
//     session.conversationHistory.push({ lang, text: finalText });
//     if (session.conversationHistory.length > 20)
//       session.conversationHistory.shift();

//     this.resetBuffer(buf);
//     this.claudePipeline(finalText, lang, session, sessionId);
//   }

//   // ─── Transcripción en tiempo real ─────────────────────────────────────────

//   async startRealTimeTranscription(
//     sessionId: string,
//     callback: (data: string) => void,
//   ): Promise<{ send: (chunk: ArrayBuffer) => void; close: () => void }> {
//     const apiKey = process.env.ASSEMBLYAI_API_KEY;
//     if (!apiKey) throw new Error('ASSEMBLYAI_API_KEY no configurada');

//     const session: SessionData = {
//       buffer: this.emptyBuf(),
//       conversationHistory: [],
//       chunkCount: 0,
//       callback,
//     };
//     this.sessionData.set(sessionId, session);
//     this.logger.log(`🎤 AssemblyAI v3 iniciando [${sessionId}]`);

//     const params = new URLSearchParams({
//       sample_rate: '16000',
//       format_turns: 'true',
//       speech_model: 'universal-streaming-multilingual',
//       language_detection: 'true',
//       end_of_turn_confidence_threshold: '0.6',
//       max_turn_silence: '1000',
//     });

//     const KEYTERMS = [
//       'Keppra',
//       'convulsión',
//       'convulsiones',
//       'epilepsia',
//       'seizure',
//       'seizures',
//       'levetiracetam',
//       'medicamento',
//       'medicamentos',
//       'valproato',
//       'carbamazepina',
//       'lamotrigina',
//       'cerebro',
//       'dosis',
//     ];

//     const WebSocket = require('ws');
//     const ws = new WebSocket(
//       `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`,
//       { headers: { Authorization: apiKey } },
//     );

//     ws.on('open', () =>
//       this.logger.log(`✅ AssemblyAI v3 abierto [${sessionId}]`),
//     );
//     ws.on('error', (err: Error) =>
//       this.logger.error(
//         `❌ AssemblyAI v3 error [${sessionId}]: ${err.message}`,
//       ),
//     );

//     ws.on('close', (code: number) => {
//       this.logger.log(`🔒 AssemblyAI v3 cerrado [${sessionId}] (${code})`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text) this.closeTurn(sessionId, 'streamClose');
//       this.sessionData.delete(sessionId);
//     });

//     ws.on('message', (raw: any) => {
//       const s = this.sessionData.get(sessionId);
//       if (!s) return;
//       let msg: any;
//       try {
//         msg = JSON.parse(raw.toString());
//       } catch {
//         return;
//       }

//       const buf = s.buffer;
//       const now = Date.now();

//       if (msg.type === 'Begin') {
//         this.logger.log(`🔗 Sesión AssemblyAI v3 [${sessionId}] sid=${msg.id}`);
//         ws.send(
//           JSON.stringify({ type: 'UpdateConfiguration', keyterms: KEYTERMS }),
//         );
//         this.logger.log(`📚 Keyterms enviados [${sessionId}]`);
//         return;
//       }

//       if (msg.type === 'Turn') {
//         const text: string = (msg.transcript || '').trim();
//         const aaiLang: string = msg.language_code;
//         const aaiConf: number = msg.language_confidence ?? 0;
//         const isFinal: boolean = msg.turn_is_formatted === true;
//         const wordCount = text.split(/\s+/).filter(Boolean).length;

//         this.logger.log(
//           `🔬 RAW [${sessionId}] fmt=${isFinal} lang=${aaiLang} conf=${aaiConf.toFixed(2)} text="${text.substring(0, 60)}"`,
//         );
//         if (!text) return;

//         // ── Filtro de ruido: rechazar si AAI detectó idioma != es/en con conf baja ─
//         // Ruido ambiental produce transcripciones en fr/it/pt con conf < 0.35
//         // y texto corto (1-2 palabras). Ejemplo: lang=fr conf=0.59 text="Conditions."
//         // EXCEPCIÓN: palabras universales como "No/Si/Sí/Yes/Ok" no se descartan
//         // porque son válidas en múltiples idiomas y son respuestas médicas importantes.
//         const isNonTargetLang =
//           aaiLang &&
//           aaiLang !== 'en' &&
//           aaiLang !== 'es' &&
//           aaiLang !== 'undefined';
//         const isUniversalWord = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(
//           text.trim(),
//         );
//         if (
//           isNonTargetLang &&
//           aaiConf < 0.65 &&
//           wordCount <= 2 &&
//           !isUniversalWord
//         ) {
//           this.logger.log(
//             `🚫 Ruido descartado [${aaiLang}=${aaiConf.toFixed(2)}] "${text}" [${sessionId}]`,
//           );
//           return;
//         }

//         // ── Guard de continuación post-close ─────────────────────────────────
//         // AAI v3 sigue emitiendo Turns del mismo utterance después de que el
//         // silence timer ya cerró el bloque. Si el buffer está vacío, se cerró
//         // hace < 1200ms, y el texto nuevo empieza con los primeros ~20 chars
//         // del bloque anterior → es continuación. Reabrimos el buffer en silencio.
//         //
//         // EXCEPCIÓN CRÍTICA: si el cierre fue un ForceClose por mezcla EN+ES,
//         // NO reabrir — AAI sigue enviando el mismo Turn fusionado y si reabrimos
//         // volvemos a acumular texto mezclado. Bloqueamos por 2000ms post-ForceClose.
//         const msSinceClose = now - buf.lastClosedMs;
//         const msSinceForceClose = now - buf.forceClosedMs;
//         const forceCloseBlackout = msSinceForceClose < 2000;
//         if (
//           !buf.text &&
//           msSinceClose < 1200 &&
//           buf.lastEmittedText &&
//           !forceCloseBlackout
//         ) {
//           const normalize = (s: string) =>
//             s
//               .replace(/Keppra/gi, 'kepra')
//               .replace(/[,\.!?¿¡]/g, '')
//               .replace(/\s+/g, ' ')
//               .trim()
//               .toLowerCase();
//           const prevNorm = normalize(buf.lastEmittedText);
//           const curNorm = normalize(text);
//           const prefix = prevNorm.substring(0, Math.min(prevNorm.length, 20));
//           if (prefix.length >= 4 && curNorm.startsWith(prefix)) {
//             this.logger.log(
//               `🔁 ContinuationGuard reopen [${sessionId}] +${msSinceClose}ms`,
//             );
//             buf.text = text;
//             buf.lang = buf.lastEmittedLang;
//             this.clearTimer(buf);
//             buf.timer = setTimeout(() => {
//               buf.timer = null;
//               this.logger.log(`⏱ Silence close [${sessionId}]`);
//               this.closeTurn(sessionId, 'silence');
//             }, T_SILENCE_CLOSE);
//             return;
//           }
//         }

//         // Detectar idioma con señal de fuerza léxica
//         const { lang: lexLang, strong: lexStrong } =
//           this.detectLangWithStrength(text);
//         const detectedLang = this.resolveLang(
//           text,
//           aaiLang,
//           aaiConf,
//           buf.lang,
//           wordCount,
//         );
//         if (aaiLang)
//           this.logger.log(
//             `🌐 ASR lang=${aaiLang} conf=${aaiConf.toFixed(3)} words=${wordCount} → ${detectedLang} (lex=${lexLang} strong=${lexStrong})`,
//           );

//         // ── Asignar idioma al buffer ────────────────────────────────────
//         const bufEmpty = !buf.lang || !buf.text;
//         if (bufEmpty) {
//           if (
//             lexStrong &&
//             buf.lastEmittedLang &&
//             buf.lastEmittedLang !== lexLang
//           ) {
//             // Caso B: léxico fuerte señala idioma diferente al turno previo.
//             buf.lang = lexLang;
//             this.logger.log(
//               `🌍 LangFromLex [${buf.lastEmittedLang}→${lexLang}] post-close [${sessionId}]`,
//             );
//           } else if (
//             !lexStrong &&
//             this.isBackchannel(text) &&
//             buf.lastEmittedLang
//           ) {
//             // Caso C: backchannel ambiguo (No/Sí/Ok/4...) sin señal léxica fuerte.
//             // Asumir idioma CONTRARIO al turno anterior (doctor EN → paciente ES y viceversa).
//             // EXCEPCIÓN: Si el backchannel es "Si/Sí/No" y el turno anterior fue ES,
//             // NO invertir — es más probable que el paciente continúe en ES que el doctor
//             // diga "see?" o "no?" en ese momento. En ese caso mantener ES.
//             const isSpanishBackchannel = /^(sí|si|no)\.?,?$/i.test(text.trim());
//             if (isSpanishBackchannel && buf.lastEmittedLang === 'es') {
//               buf.lang = 'es';
//               this.logger.log(
//                 `🔄 BackchanelKeep [es] text="${text}" [${sessionId}]`,
//               );
//             } else {
//               const opposite = buf.lastEmittedLang === 'en' ? 'es' : 'en';
//               buf.lang = opposite;
//               this.logger.log(
//                 `🔄 BackchanelFlip [${buf.lastEmittedLang}→${opposite}] text="${text}" [${sessionId}]`,
//               );
//             }
//           } else {
//             buf.lang = detectedLang;
//           }
//         } else if (aaiLang && aaiConf > 0.8) {
//           buf.lang = detectedLang;
//         }

//         // ── Speaker change ──────────────────────────────────────────────
//         // REGLA CRÍTICA: solo disparar si hay un silencio real entre hablantes.
//         // Si el texto del buffer sigue creciendo activamente (lastUpdateMs reciente),
//         // es el MISMO hablante — no importa si AAI cambia su estimación de idioma
//         // a mitad de un utterance. "Keppra Y lo dejé..." empieza como EN y luego
//         // AAI lo reclasifica como ES — sin silencio entre medio, no es speaker change.
//         //
//         // GUARD GEOMÉTRICO: si el texto nuevo EMPIEZA con el texto del buffer,
//         // es el mismo hablante creciendo — imposible que sea speaker change.
//         // "Keppra Y" → "Keppra Y lo" → startsWith → mismo turno, sin cambio.
//         const isGrowingTurn = buf.text && text.startsWith(buf.text.trimEnd());

//         // Condiciones para speaker change (TODAS deben cumplirse):
//         // 1. El texto no está creciendo (guard geométrico)
//         // 2. Silencio real: > 400ms desde el último Turn event
//         // 3. Idioma detectado con confianza alta
//         const silenceGap = now - buf.lastUpdateMs > 400;
//         const confOk = aaiConf >= MIN_SPEAKER_CHANGE_CONF && wordCount >= 2;
//         const veryConf = aaiConf >= 0.8;
//         const lexConfChange =
//           lexStrong && buf.lang && buf.lang !== lexLang && buf.text;
//         const bufLangChanged =
//           buf.lang && buf.lang !== detectedLang && buf.text;

//         if (
//           !isGrowingTurn &&
//           silenceGap &&
//           ((bufLangChanged && (confOk || veryConf)) ||
//             (lexConfChange && wordCount >= 3))
//         ) {
//           this.logger.log(
//             `🔀 SpeakerChange [${buf.lang}→${detectedLang}] gap=${now - buf.lastUpdateMs}ms [${sessionId}]`,
//           );
//           this.closeTurn(sessionId, 'speakerChange');
//           buf.lang = detectedLang;
//         }

//         buf.lastUpdateMs = now;

//         // ── Acumular texto en buffer + emitir partial en vivo ──────────
//         buf.text = text;
//         this.emitPartial(s, sessionId);
//         this.logger.log(
//           `📝 ${isFinal ? 'FINAL' : 'Partial'} [${buf.lang}] [${sessionId}]: "${text.substring(0, 80)}"`,
//         );

//         // ── ForceClose por mezcla de idiomas en mismo Turn ──────────────
//         // Cuando AAI fusiona doctor+paciente en un mismo Turn, el texto
//         // acumulado contiene frases EN seguidas de frases ES (o viceversa).
//         // Al detectar mezcla con ≥8 palabras, cerramos INMEDIATAMENTE y
//         // retornamos para que el silence timer normal no sobreescriba.
//         if (wordCount >= 8 && buf.text) {
//           const words = text.trim().split(/\s+/);
//           const esOnlyWords =
//             /^(que|los|las|del|una|con|para|pero|desde|hace|porque|también|cuando|como|esto|eso|fue|han|tengo|tuve|tenía|convulsiones|días|mes|año|años|siempre|nunca|alguna|dejé|pagar|cobraba|incrementaron|tomarla|todos)$/i;
//           const enOnlyWords =
//             /^(the|and|you|have|had|are|taking|medications|seizures|since|before|after|dose|increase|missed|those|pills|times|every|medical|conditions|family|history|examine|when|was|your|last|seizure|not)$/i;
//           const lastThird = words.slice(Math.floor(words.length * 0.6));
//           const firstHalf = words.slice(0, Math.floor(words.length * 0.5));
//           const firstHasEN = firstHalf.some((w) => enOnlyWords.test(w));
//           const firstHasES = firstHalf.some((w) => esOnlyWords.test(w));
//           const lastHasEN = lastThird.some((w) => enOnlyWords.test(w));
//           const lastHasES = lastThird.some((w) => esOnlyWords.test(w));
//           const mixDetected =
//             (firstHasEN && lastHasES) || (firstHasES && lastHasEN);
//           if (mixDetected) {
//             this.logger.log(
//               `🔀 ForceClose por mezcla EN+ES [${sessionId}] "${text.substring(0, 60)}"`,
//             );
//             this.clearTimer(buf);
//             buf.forceClosedMs = now; // bloquear ContinuationGuard para este Turn de AAI
//             this.closeTurn(sessionId, 'silence'); // cierre síncrono inmediato
//             return; // no continuar al silence timer — ya cerramos
//           }
//         }
//         // ── Silence timer: detectar Turn estancado ──────────────────────
//         // Con end_of_turn_confidence_threshold=1.0, AAI nunca cierra su Turn
//         // propio. Cuando el speaker hace pausa, AAI sigue enviando el MISMO
//         // texto repetidamente (el buffer del Turn está "congelado"). En ese
//         // caso NO resetear el timer — dejar que expire para cerrar el bloque.
//         // Cuando el texto SÍ crece (nueva speech), resetar normalmente.
//         const textGrew = text !== buf.lastSeenText;
//         buf.lastSeenText = text;
//         if (textGrew) {
//           buf.staleCount = 0;
//           // Texto nuevo → resetear timer
//           this.clearTimer(buf);
//           buf.timer = setTimeout(() => {
//             buf.timer = null;
//             this.logger.log(`⏱ Silence close [${sessionId}]`);
//             this.closeTurn(sessionId, 'silence');
//           }, T_SILENCE_CLOSE);
//         } else {
//           buf.staleCount++;
//           // Texto estancado (mismo que antes) → NO resetear timer, dejar que expire
//           // Loguear solo ocasionalmente para no saturar
//           if (buf.staleCount === 3) {
//             this.logger.log(
//               `🧊 Turn estancado [${sessionId}] stale=${buf.staleCount} — timer no reseteado`,
//             );
//           }
//           // Si no hay timer activo (fue limpiado), crear uno nuevo de todas formas
//           if (!buf.timer) {
//             buf.timer = setTimeout(() => {
//               buf.timer = null;
//               this.logger.log(`⏱ Silence close [${sessionId}]`);
//               this.closeTurn(sessionId, 'silence');
//             }, T_SILENCE_CLOSE);
//           }
//         }
//       } else if (msg.type === 'Termination') {
//         this.logger.log(
//           `🏁 Terminado [${sessionId}] audio=${msg.audio_duration_seconds}s`,
//         );
//       }
//     });

//     const send = (chunk: ArrayBuffer) => {
//       const s = this.sessionData.get(sessionId);
//       if (!s) return;
//       s.chunkCount++;
//       if (s.chunkCount % 40 === 0)
//         this.logger.log(`📤 [${sessionId}] Chunk #${s.chunkCount}`);
//       if (ws.readyState === WebSocket.OPEN) ws.send(chunk);
//     };

//     const close = async () => {
//       this.logger.log(`⏳ Cerrando AssemblyAI v3 [${sessionId}]`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text) await this.closeTurn(sessionId, 'userStop');
//       if (ws.readyState === WebSocket.OPEN) {
//         ws.send(JSON.stringify({ type: 'Terminate' }));
//         // Esperar más tiempo para que AAI procese el audio en buffer antes de cerrar.
//         // Con 800ms se perdían las últimas frases del doctor. Con 2500ms damos tiempo
//         // suficiente para que AAI emita los Turns pendientes y los procesemos.
//         await new Promise((r) => setTimeout(r, 2500));
//       }
//       ws.close();
//       this.logger.log(`🛑 AssemblyAI v3 cerrado [${sessionId}]`);
//     };

//     return { send, close };
//   }

//   // ─── Claude (background, no bloquea display) ──────────────────────────────

//   private async claudePipeline(
//     text: string,
//     lang: 'es' | 'en',
//     session: SessionData,
//     sessionId: string,
//   ) {
//     const history = [...session.conversationHistory];
//     const { result, correctedLang } = await this.correctWithClaude(
//       text,
//       lang,
//       history,
//     );
//     if (result !== text || correctedLang !== lang) {
//       this.logger.log(
//         `✨ CLAUDE [${lang}→${correctedLang}]: "${result.substring(0, 80)}"`,
//       );
//       const idx = session.conversationHistory.findLastIndex(
//         (t) => t.text === text,
//       );
//       if (idx >= 0) {
//         session.conversationHistory[idx].text = result;
//         session.conversationHistory[idx].lang = correctedLang;
//       }
//       // Si Claude corrigió el idioma, actualizar lastEmittedLang para que
//       // el ContinuationGuard y BackchanelFlip usen el idioma correcto
//       if (correctedLang !== lang && session.buffer.lastEmittedLang === lang) {
//         session.buffer.lastEmittedLang = correctedLang;
//       }
//       this.emit(session, {
//         text: result,
//         language: correctedLang,
//         isCorrection: true,
//         originalText: text,
//         sessionId,
//       });
//     }
//   }

//   private async correctWithClaude(
//     text: string,
//     lang: 'es' | 'en',
//     history: ConversationTurn[],
//   ): Promise<{ result: string; correctedLang: 'es' | 'en' }> {
//     if (!this.anthropic || text.length < 5)
//       return { result: text, correctedLang: lang };
//     const ctx = history
//       .slice(0, -1)
//       .slice(-5)
//       .map((t) => `[${t.lang === 'en' ? 'Doctor' : 'Patient'}]: ${t.text}`)
//       .join('\n');

//     const prompt = `You are an ASR post-processor for a bilingual medical interpreter. Doctor speaks English, Patient speaks Spanish.
// ${ctx ? `Conversation so far:\n${ctx}\n` : ''}
// ASR transcription to fix: "${text}"
// Detected language: ${lang === 'es' ? 'Spanish (patient)' : 'English (doctor)'}

// RULES — apply ONLY these corrections:
// 1. "kepra/keprah/kephra/quepra/kebra" → "Keppra"
// 2. Spanish "see " or "si " at utterance start → "Sí, "
// 3. "2000" in dosage context → "2,000"
// 4. Clear phonetic errors: "Wer you" → "Were you", "hav you" → "have you"
// 5. Fix obvious punctuation only
// 6. DO NOT add words, DO NOT complete sentences, DO NOT translate
// 7. If nothing to fix, return text EXACTLY as-is
// 8. CRITICAL — Wrong language detection: If the detected language is English but the text looks like garbled Spanish (e.g. "See those mean" could be "Si dos mil", "See" could be "Sí"), AND the conversation context shows the patient was just speaking Spanish about dosages, correct it to the most likely Spanish. Example: after patient says dosage info in Spanish, "See those mean." → "Sí, dos mil."

// Output ONLY the corrected text — no explanations, no quotes.`;

//     try {
//       const r = await this.anthropic.messages.create({
//         model: 'claude-haiku-4-5-20251001',
//         max_tokens: 300,
//         messages: [{ role: 'user', content: prompt }],
//       });
//       const result = (r.content[0] as any).text?.trim() || text;
//       if (this.norm(result) === this.norm(text))
//         return { result: text, correctedLang: lang };
//       if (result.length > text.length * 1.4 + 20)
//         return { result: text, correctedLang: lang };
//       // Detectar si Claude corrigió el idioma (ej: "See those mean" → "Sí, dos mil")
//       const detectedResultLang = this.detectLang(result);
//       const correctedLang: 'es' | 'en' = detectedResultLang ?? lang;
//       return { result, correctedLang };
//     } catch (e: any) {
//       this.logger.error(`❌ Claude correct: ${e.message}`);
//       return { result: text, correctedLang: lang };
//     }
//   }
// }
