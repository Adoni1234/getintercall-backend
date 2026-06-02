import { Injectable, Logger } from '@nestjs/common';
import { AssemblyAI } from 'assemblyai';
import Anthropic from '@anthropic-ai/sdk';

const T_SILENCE_CLOSE = 1200;
const T_SILENCE_CLOSE_STALE = 600;
const MIN_SPEAKER_CHANGE_CONF = 0.72;

interface TurnBuffer {
  text: string;
  lang: 'es' | 'en' | null;
  lastUpdateMs: number;
  lastClosedMs: number;
  lastEmittedText: string;
  lastEmittedLang: 'es' | 'en' | null;
  timer: NodeJS.Timeout | null;
  lastSeenText: string;
  staleCount: number;
  forceClosedMs: number;
  peakText: string;
  langConfident: boolean;
  // Debounce de partials — timestamp del último partial emitido
  lastPartialEmitMs: number;
}

interface ConversationTurn {
  lang: 'es' | 'en';
  text: string;
}

interface SessionData {
  buffer: TurnBuffer;
  conversationHistory: ConversationTurn[];
  chunkCount: number;
  callback: (data: string) => void;
  ws?: any;
}

@Injectable()
export class TranscribeService {
  private readonly logger = new Logger(TranscribeService.name);
  private assembly: AssemblyAI | null = null;
  private anthropic: Anthropic | null = null;
  private sessionData = new Map<string, SessionData>();

  constructor() {
    const assemblyKey = process.env.ASSEMBLYAI_API_KEY;
    const claudeKey = process.env.ANTHROPIC_API_KEY;
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
      audio: file.buffer,
      language_code: 'es',
    });
    return { text: t.text || '' };
  }

  private emptyBuf(): TurnBuffer {
    return {
      text: '',
      lang: null,
      lastUpdateMs: 0,
      lastClosedMs: 0,
      forceClosedMs: 0,
      lastEmittedText: '',
      lastEmittedLang: null,
      timer: null,
      lastSeenText: '',
      staleCount: 0,
      peakText: '',
      langConfident: false,
      lastPartialEmitMs: 0,
    };
  }

  private clearTimer(buf: TurnBuffer) {
    if (buf.timer) { clearTimeout(buf.timer); buf.timer = null; }
  }

  private resetBuffer(buf: TurnBuffer) {
    buf.text = '';
    buf.lang = null;
    buf.lastUpdateMs = 0;
    buf.timer = null;
    buf.lastSeenText = '';
    buf.staleCount = 0;
    buf.peakText = '';
    buf.langConfident = false;
    buf.lastPartialEmitMs = 0;
  }

  private detectLang(text: string): 'es' | 'en' {
    const t = text.toLowerCase();
    const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar)\b/g) || []).length;
    const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes)\b/g) || []).length;
    return esScore > enScore ? 'es' : 'en';
  }

  private detectLangWithStrength(text: string): { lang: 'es' | 'en'; strong: boolean } {
    const t = text.toLowerCase();
    const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar)\b/g) || []).length;
    const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes)\b/g) || []).length;
    const lang = esScore > enScore ? 'es' : 'en';
    const strong = Math.max(esScore, enScore) >= 2 || Math.abs(esScore - enScore) >= 2;
    return { lang, strong };
  }

  private resolveLang(
    text: string, aaiLang: string | undefined, aaiConf: number,
    bufLang: 'es' | 'en' | null, wordCount: number,
  ): 'es' | 'en' {
    if (aaiLang && aaiConf > 0.40 && aaiLang !== 'undefined') {
      return aaiLang.startsWith('es') ? 'es' : 'en';
    }
    const { lang: lexLang, strong } = this.detectLangWithStrength(text);
    if (strong) return lexLang;
    if (wordCount <= 2 && bufLang) return bufLang;
    return lexLang;
  }

  private fixText(text: string, lang: 'es' | 'en'): string {
    let t = text.trim();
    t = t.replace(/\b(keprah?|kepra|quepra|kephra|kebri[ah]?|kebra)\b/gi, 'Keppra');
    if (lang === 'es') t = t.replace(/^(see|si)\s/i, 'Sí, ').replace(/\b2[\s,]?000\b/g, '2,000');
    if (lang === 'en') t = t.replace(/\b2[\s,]?000\b/g, '2,000');
    const enStartWords = /^(or|before|after|the|was|were|is|are|have|had|do|does|did|when|where|what|how|why|which|that|this|it|in|of|for|with|a|an|and|but|not|no|any|all|one|two|three|four|some|your|their|our|my|its)/i;
    const shortPrefixMatch = t.match(/^(\d{1,3}\.?\s+)(\w.+)/);
    if (shortPrefixMatch && enStartWords.test(shortPrefixMatch[2])) {
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

  private emit(session: SessionData, payload: object) {
    session.callback(JSON.stringify(payload));
  }

  private emitPartial(session: SessionData, sessionId: string) {
    const buf = session.buffer;
    if (!buf.text || !buf.lang) return;
    const words = buf.text.trim().split(/\s+/).filter(Boolean).length;
    const isKnownBackchannel = /^(sí|si|no|yes|ok|yeah|cuatro|four|tres|three|dos|two|uno|one|bien|claro)\.?,?$/i.test(buf.text.trim());
    if (words < 2 && !isKnownBackchannel) return;

    // Debounce: no emitir partials más rápido que cada 150ms
    const now = Date.now();
    if (now - buf.lastPartialEmitMs < 150) return;
    buf.lastPartialEmitMs = now;

    this.emit(session, { text: buf.text, language: buf.lang, isNewTurn: false, sessionId });
  }

  private async closeTurn(sessionId: string, reason: string): Promise<void> {
    const session = this.sessionData.get(sessionId);
    if (!session) return;
    const buf = session.buffer;
    const textToClose = (buf.peakText && buf.peakText.length > (buf.text?.length || 0))
      ? buf.peakText : buf.text;
    if (!textToClose) return;

    this.clearTimer(buf);
    const lang = buf.lang ?? this.detectLang(textToClose);
    const finalText = this.fixText(textToClose, lang);
    if (!finalText) { this.resetBuffer(buf); return; }

    const wordCount = finalText.trim().split(/\s+/).length;
    const isUniversalBackchannel = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(finalText.trim());
    const isNumericResponse = /^\d+\.?$/.test(finalText.trim());

    // Filtro de eco: palabra única que ya aparece al final de un bloque reciente
    if (wordCount === 1 && !isUniversalBackchannel && !isNumericResponse) {
      const w = this.norm(finalText);
      const prev = this.norm(buf.lastEmittedText ?? '');
      if (prev.endsWith(w)) { this.resetBuffer(buf); return; }
      const recentHistory = session.conversationHistory.slice(-5);
      for (const h of recentHistory) {
        if (this.norm(h.text).endsWith(w)) {
          this.logger.log(`🔇 Eco [${sessionId}]: "${finalText}"`);
          this.resetBuffer(buf);
          return;
        }
      }
    }

    const isShortBackchannel = wordCount <= 2;
    if (!isShortBackchannel && this.norm(finalText) === this.norm(buf.lastEmittedText)) {
      this.logger.log(`⏭ Dedup [${lang}] [${sessionId}]`);
      this.resetBuffer(buf);
      return;
    }

    // Filtro de ruido de 1 char
    if (finalText.trim().length === 1 && !isUniversalBackchannel && !isNumericResponse) {
      this.logger.log(`🚫 Ruido 1-char [${sessionId}]: "${finalText}"`);
      this.resetBuffer(buf);
      return;
    }

    this.logger.log(`✅ CLOSE [${lang}] [${sessionId}] (${reason}): "${finalText.substring(0, 80)}"`);
    buf.lastEmittedText = finalText;
    buf.lastEmittedLang = lang;
    buf.lastClosedMs = Date.now();

    this.emit(session, { text: finalText, language: lang, isNewTurn: true, isForcedClose: false, sessionId });
    session.conversationHistory.push({ lang, text: finalText });
    if (session.conversationHistory.length > 20) session.conversationHistory.shift();

    this.resetBuffer(buf);
    this.claudePipeline(finalText, lang, session, sessionId);
  }

  async startRealTimeTranscription(
    sessionId: string,
    callback: (data: string) => void,
  ): Promise<{ send: (chunk: ArrayBuffer) => void; close: () => void }> {
    const apiKey = process.env.ASSEMBLYAI_API_KEY;
    if (!apiKey) throw new Error('ASSEMBLYAI_API_KEY no configurada');

    const session: SessionData = {
      buffer: this.emptyBuf(),
      conversationHistory: [],
      chunkCount: 0,
      callback,
    };
    this.sessionData.set(sessionId, session);
    this.logger.log(`🎤 AssemblyAI v3 iniciando [${sessionId}]`);

    // u3-rt-pro: modelo recomendado por AssemblyAI.
    // Detecta fin de turno por puntuación (no por confianza), code-switching
    // EN/ES nativo sin parámetros extra, sub-300ms.
    const params = new URLSearchParams({
      sample_rate: '16000',
      format_turns: 'true',
      speech_model: 'u3-rt-pro',
    });

    const U3_PROMPT = 'Bilingual medical interpreter conversation. Doctor speaks English, patient speaks Spanish. Medical terminology includes seizures, Keppra, epilepsy, convulsiones, medicamentos.';

    const KEYTERMS = [
      'Keppra', 'convulsión', 'convulsiones', 'epilepsia',
      'seizure', 'seizures', 'levetiracetam', 'medicamento',
      'medicamentos', 'valproato', 'carbamazepina', 'lamotrigina',
      'cerebro', 'dosis',
    ];

    const WebSocket = require('ws');
    const ws = new WebSocket(
      `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`,
      { headers: { Authorization: apiKey } },
    );

    session.ws = ws;
    ws.on('open', () => this.logger.log(`✅ AssemblyAI v3 abierto [${sessionId}]`));
    ws.on('error', (err: Error) => this.logger.error(`❌ AAI error [${sessionId}]: ${err.message}`));

    ws.on('close', (code: number) => {
      this.logger.log(`🔒 AAI cerrado [${sessionId}] (${code})`);
      const s = this.sessionData.get(sessionId);
      if (s?.buffer.text || s?.buffer.peakText) this.closeTurn(sessionId, 'streamClose');
      this.sessionData.delete(sessionId);
    });

    ws.on('message', async (raw: any) => {
      const s = this.sessionData.get(sessionId);
      if (!s) return;
      let msg: any;
      try { msg = JSON.parse(raw.toString()); } catch { return; }

      const buf = s.buffer;
      const now = Date.now();

      if (msg.type === 'Begin') {
        this.logger.log(`🔗 AAI v3 [${sessionId}] sid=${msg.id}`);
        ws.send(JSON.stringify({ type: 'UpdateConfiguration', keyterms: KEYTERMS, prompt: U3_PROMPT }));
        return;
      }

      if (msg.type === 'Turn') {
        const text: string = (msg.transcript || '').trim();
        const aaiLang: string = msg.language_code ?? '';
        const aaiConf: number = msg.language_confidence ?? 0;
        const isFinal: boolean = msg.turn_is_formatted === true;
        const wordCount = text.split(/\s+/).filter(Boolean).length;

        const hasRealLang = !!(aaiLang && aaiLang !== 'undefined' && aaiConf > 0);

        const isUniversalWord = /^(no|sí|si|yes|ok|yeah|bien|\d+)\.?,?$/i.test(text.trim());

        if (isFinal && (s as any)._forceEndpointFallback) {
          clearTimeout((s as any)._forceEndpointFallback);
          delete (s as any)._forceEndpointFallback;
        }

        this.logger.log(`🔬 RAW fmt=${isFinal} lang=${aaiLang} conf=${aaiConf.toFixed(2)} "${text.substring(0, 60)}" [${sessionId}]`);
        if (!text) return;

        // ── Filtro de ruido ────────────────────────────────────────────────────
        const isNonTargetLang = hasRealLang && aaiLang !== 'en' && aaiLang !== 'es';
        if (isNonTargetLang && aaiConf < 0.65 && wordCount <= 2 && !isUniversalWord) {
          this.logger.log(`🚫 Ruido [${aaiLang}=${aaiConf.toFixed(2)}] "${text}" [${sessionId}]`);
          return;
        }

        // ── FIX #5: Filtros de ruido avanzados ───────────────────────────────
        //
        // TIPO A — Artefactos de inicio de turno (buffer vacío, turno corto malformado)
        if (isFinal && wordCount >= 2 && wordCount <= 4 && !buf.text) {
          const tNorm = text.toLowerCase().replace(/[¿?!¡.,]/g, '').trim();
          const knownArtifacts = [
            'se despierta', 'qué hace como', 'es eso cómo', 'eso cómo',
            'hace como', 'es eso', 'as you may know', 'señor', 'i see him',
            'cómo hay', 'qué qué', 'qué es eso',
          ];
          const isKnownArtifact = knownArtifacts.some(a => tNorm === a);
          // Pregunta muy corta sin vocabulario médico/conversacional real
          const medicalOrCommon = /\b(convulsión|convulsiones|keppra|seizure|seizures|epilepsia|medicamento|dosis|dolor|espalda|cabeza|cerebro|hospital|doctor|médico|why|here|have|had|taking|your|you|when|last|how|many|what|were|before|after|increase|dose|missed|ever|day|days|sí|no|yes|okay|because|pero|desde|hace|tengo|tiene|tuve|dejé|pagar|cobrar|todos|días|años|meses)\b/i;
          const isGenericArtifact = wordCount <= 3 && text.endsWith('?') && !medicalOrCommon.test(text);
          if (isKnownArtifact || isGenericArtifact) {
            this.logger.log(`🗑️ Artefacto inicio [${sessionId}]: "${text}"`);
            return;
          }
        }

        // TIPO B — Bloques de subtítulos/overlay del video o intérprete repetitivo
        // Patrón: múltiples frases cortas separadas por punto en una sola línea,
        // o la misma palabra/frase repetida dos veces (ej: "¿Qué es Keppra? Epilepsia. ¿Qué es Keppra?")
        // o lista de términos médicos sin verbo (ej: "Seizures. Epilepsy. Convulsiones.")
        if (isFinal) {
          const sentences = text.split(/[.!?¿¡]+/).map(s => s.trim()).filter(Boolean);
          // Detectar repetición: la misma frase aparece 2+ veces
          const normalized = sentences.map(s => s.toLowerCase().replace(/\s+/g, ' ').trim());
          const hasDuplicateSentence = new Set(normalized).size < normalized.length && sentences.length >= 2;
          // Detectar lista de términos médicos sin verbo ni sujeto real
          // (todos los fragmentos son ≤2 palabras y son términos médicos puros)
          const isMedicalList = sentences.length >= 3 && sentences.every(s => {
            const words = s.trim().split(/\s+/).filter(Boolean);
            return words.length <= 2 && /\b(seizure|seizures|epilepsy|epilepsia|convulsión|convulsiones|keppra|medication|medicamento)\b/i.test(s);
          });
          if (hasDuplicateSentence || isMedicalList) {
            this.logger.log(`🗑️ Subtítulo/overlay [${sessionId}]: "${text.substring(0, 60)}"`);
            return;
          }
        }

        // TIPO C — Fragmentos incompletos y bloques de ruido cortos
        // Casos cubiertos:
        // 1. Sin puntuación final, ≤4 palabras → esperar continuación silenciosamente
        // 2. Termina en guión (—) → el hablante fue interrumpido, suprimir
        // 3. Palabra única médica sin contexto ("Seizures.", "Keppra.") → suprimir
        //    (son ecos fonéticos o audio del sistema, no turnos reales)
        const endsWithDash = /[—–-]{1,2}$/.test(text.trim());
        const hasTerminalPunct = /[.!?]$/.test(text.trim());
        const isSingleMedicalWord = wordCount === 1 && /^(seizures?|epilepsy|epilepsia|convulsiones?|keppra|medication|medicamentos?)\.?$/i.test(text.trim());

        // Suprimir directamente: guión final o palabra médica suelta
        if (isFinal && (endsWithDash || isSingleMedicalWord) && !buf.text) {
          this.logger.log(`🗑️ Ruido suprimido [${sessionId}]: "${text}"`);
          return;
        }

        const isIncompleteFragment = isFinal && !hasTerminalPunct && !endsWithDash && wordCount <= 4 && !buf.text;
        if (isIncompleteFragment) {
          this.logger.log(`⏳ Fragmento incompleto silencioso [${sessionId}]: "${text}"`);
          buf.text = text;
          buf.peakText = text;
          buf.lang = this.resolveLang(text, aaiLang, aaiConf, buf.lastEmittedLang, wordCount);
          buf.langConfident = hasRealLang;
          buf.lastUpdateMs = now;
          buf.lastSeenText = text;
          buf.lastClosedMs = now;
          this.clearTimer(buf);
          buf.timer = setTimeout(() => {
            buf.timer = null;
            const sCheck = this.sessionData.get(sessionId);
            if (sCheck?.buffer.text === text) {
              this.logger.log(`🗑 Fragmento suprimido [${sessionId}]: "${text}"`); if (sCheck) this.resetBuffer(sCheck.buffer);
            }
          }, 1200);
          return;
        }

        // CORRECCIÓN DE IDIOMA: texto en español asignado a [en] por error de AAI
        // Cuando u3-rt-pro no devuelve lang (conf=0.00) y el texto tiene léxico ES
        // fuerte, pero el buffer está vacío y buf.lastEmittedLang es 'en', AAI a veces
        // asigna ES al doctor. Detectar y corregir antes de procesar.
        // Ejemplos: "Eso fue antes.", "Sí, doctor.", "Sí, dos mil."
        if (isFinal && !hasRealLang && !buf.text) {
          const { lang: detectedLex, strong: lexStrongHere } = this.detectLangWithStrength(text);
          const msSinceLast = now - buf.lastClosedMs;
          // Si léxico ES fuerte Y el último bloque emitido fue también ES (es respuesta del paciente)
          // O si el texto empieza con "Sí"/"Eso"/"Hace" (español obvio) y está mal asignado
          const startsObviouslyES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|y |no,)/i.test(text.trim());
          if ((lexStrongHere && detectedLex === 'es') || startsObviouslyES) {
            // Forzar ES independientemente del lastEmittedLang
            // (el paciente responde en ES, el doctor pregunta en EN)
            this.logger.log(`🔧 LangCorrect forzado ES [${sessionId}]: "${text.substring(0, 50)}"`);
            // No return — dejar fluir con lang corregido en el bloque de asignación abajo
            // El bloque de asignación detectará lexStrong y usará LangFromLex
          }
        }

        // TIPO D — Filtro de intérprete
        // El intérprete está en la llamada y traduce EN→ES pero no debe aparecer.
        // Criterio 1: lista de patrones conocidos de logs.
        // Criterio 2: pregunta ES que llega <2s después de cerrar un bloque EN
        //   y comparte vocabulario médico con ese bloque — es traducción del intérprete.
        if (isFinal) {
          const tLow = text.toLowerCase().replace(/[¿?!¡.,]/g, '').trim();
          const interpreterPatterns = [
            /^cómo hay (muchos?|muchas?) convulsiones/,
            /^así un medicamento/,
            /^es antes o después de la dosis/,
            /^tú tienes pastillas/,
            /^qué fueron ustedes/,
            /^estás tomando keppra/,
            /^y hace cuánto tiempo tiene convulsiones/,
            /^cuándo fue su últ[io]m[ao] convulsión/,
            /^es eso lo que toma ahora/,
            /^si alguna vez (ha|has|he) dejado de tomarla/,
            /^hace cuánto tiempo tiene/,
          ];
          const isKnownInterpreter = interpreterPatterns.some(p => p.test(tLow));

          // Semántico: pregunta ES rápida (<2s) después de cerrar turno EN,
          // con vocabulario médico compartido = traducción del intérprete
          const msSinceLastClose = now - buf.lastClosedMs;
          const lastWasEn = buf.lastEmittedLang === 'en';
          const isQuickEsAfterEn = lastWasEn && msSinceLastClose < 2000 && !buf.text;
          const detectLangHere = this.resolveLang(text, aaiLang, aaiConf, buf.lastEmittedLang, wordCount);
          let isSemanticInterpreter = false;
          if (isQuickEsAfterEn && detectLangHere === 'es' && wordCount <= 10 && /[?]$/.test(text.trim())) {
            const medTerms = /\b(convulsión|convulsiones|seizure|seizures|keppra|medicamento|medicamentos|dosis|dose|abril|april|junio|june|pastilla|pill|tomando|taking|dejó|stopped|cuánto|cuándo|when|antes|before|después|after|aumento|increase)\b/i;
            const lastEnText = (buf.lastEmittedText || '').toLowerCase();
            isSemanticInterpreter = medTerms.test(text) && medTerms.test(lastEnText);
          }

          if (isKnownInterpreter || isSemanticInterpreter) {
            this.logger.log(`🎭 Intérprete filtrado [${sessionId}] (${isKnownInterpreter ? 'pattern' : 'semantic'} +${now - buf.lastClosedMs}ms): "${text}"`);
            return;
          }
        }

        // ── FIX #2: peakText solo crece con texto limpio ──────────────────────
        // Evita que texto mezclado momentáneo quede guardado como peak
        // y se emita al cerrar el turno aunque el texto final fuera correcto.
        const prevPeak = buf.peakText || '';
        const isCleanGrowth = text.startsWith(prevPeak.substring(0, Math.min(prevPeak.length, 15)));
        if (text.length > prevPeak.length && (isCleanGrowth || prevPeak.length < 10)) {
          buf.peakText = text;
        }

        // ── Guard de continuación post-close ──────────────────────────────────
        const msSinceClose = now - buf.lastClosedMs;
        const msSinceForceClose = now - buf.forceClosedMs;

        const normalize = (str: string) =>
          str.replace(/Keppra/gi, 'kepra').replace(/[,\.!?¿¡—–]/g, '').replace(/\s+/g, ' ').trim().toLowerCase();

        // Caso A: fragmento aún en buffer (no emitido) + llega el texto completo
        // Ej: buf="When was your" → llega "When was your last seizure?"
        // Reemplazar silenciosamente — solo emitir el texto completo
        if (buf.text && buf.text.split(/\s+/).length <= 4) {
          const bufNorm = normalize(buf.text);
          const newNorm = normalize(text);
          if (newNorm.startsWith(bufNorm.substring(0, Math.min(bufNorm.length, 12))) && newNorm.length > bufNorm.length) {
            this.logger.log(`🔁 FragmentMerge [${sessionId}]: "${buf.text}" → "${text.substring(0, 60)}"`);
            this.clearTimer(buf);
            buf.text = text;
            buf.peakText = text;
            buf.lastUpdateMs = now;
            buf.lastSeenText = text;
            // Continuar procesamiento normal con el texto completo
          }
        }

        // Caso B: turno cerrado recientemente y el nuevo texto lo extiende
        if (!buf.text && msSinceClose < 1200 && buf.lastEmittedText && msSinceForceClose >= 2000) {
          const prefix = normalize(buf.lastEmittedText).substring(0, 20);
          if (prefix.length >= 4 && normalize(text).startsWith(prefix)) {
            this.logger.log(`🔁 ContinuationGuard [${sessionId}] +${msSinceClose}ms`);
            buf.text = text;
            buf.peakText = text;
            buf.lang = buf.lastEmittedLang;
            buf.langConfident = true;
            this.clearTimer(buf);
            buf.timer = setTimeout(() => { buf.timer = null; this.closeTurn(sessionId, 'silence'); }, T_SILENCE_CLOSE);
            return;
          }
        }

        // Caso C: fragmento en buffer de idioma diferente al nuevo → flush primero
        if (buf.text && buf.text.split(/\s+/).length <= 4) {
          const bufNorm = normalize(buf.text);
          const newNorm = normalize(text);
          if (!newNorm.startsWith(bufNorm.substring(0, Math.min(bufNorm.length, 10)))) {
            this.logger.log(`🔀 FragmentFlush [${sessionId}]: "${buf.text}" → nuevo turno`);
            await this.closeTurn(sessionId, 'fragmentFlush');
          }
        }

        const { lang: lexLang, strong: lexStrong } = this.detectLangWithStrength(text);
        const silenceGap = now - buf.lastUpdateMs > 400;
        const bufEmpty = !buf.lang || !buf.text;

        // ── FIX #3: Asignación de idioma — BackchanelFlip mejorado ────────────
        //
        // "no" sin tilde es ambiguo (válido en EN y ES).
        // "sí" con tilde o "si," → definitivamente español.
        // "yes/yeah/nope" → definitivamente inglés.
        // Para "no" ambiguo: usar AAI si tiene idioma, sino mirar historial.
        // Esto evita que el doctor diga "No." y se asigne a paciente (español).

        if (isUniversalWord && buf.lastEmittedLang) {
          const isAmbiguousNo = /^no\.?,?$/i.test(text.trim());
          const isDefinitelySpanish = /^(sí|sí,|si,)$/i.test(text.trim());
          const isDefinitelyEnglish = /^(yes|yeah|nope)\.?,?$/i.test(text.trim());

          if (isDefinitelySpanish) {
            buf.lang = 'es';
            buf.langConfident = false;
            this.logger.log(`🔄 DefinitelyES "${text}" [${sessionId}]`);
          } else if (isDefinitelyEnglish) {
            buf.lang = 'en';
            buf.langConfident = false;
            this.logger.log(`🔄 DefinitelyEN "${text}" [${sessionId}]`);
          } else if (isAmbiguousNo) {
            // "No" — resolver con AAI primero, luego historial
            if (hasRealLang) {
              const noLang = aaiLang.startsWith('es') ? 'es' : 'en';
              buf.lang = noLang;
              buf.langConfident = true;
              this.logger.log(`🔄 AmbiguousNo→AAI [${noLang}] conf=${aaiConf.toFixed(2)} "${text}" [${sessionId}]`);
            } else {
              // Sin info de AAI: mirar últimos 2 bloques del historial
              // El "No" responde al último hablante, así que va al idioma contrario
              const recentLangs = s.conversationHistory.slice(-2).map(h => h.lang);
              const lastLang = recentLangs[recentLangs.length - 1] ?? buf.lastEmittedLang;
              const responseLang = lastLang === 'en' ? 'es' : 'en';
              buf.lang = responseLang;
              buf.langConfident = false;
              this.logger.log(`🔄 AmbiguousNo→History [${buf.lastEmittedLang}→${responseLang}] "${text}" [${sessionId}]`);
            }
          } else {
            // Otros universales (ok, yeah sin tilde, números) — flip normal
            const isSpanishResponse = /^(sí|si)\.?,?$/i.test(text.trim());
            if (isSpanishResponse && buf.lastEmittedLang === 'es') {
              buf.lang = 'es';
              buf.langConfident = false;
            } else {
              const opposite = buf.lastEmittedLang === 'en' ? 'es' : 'en';
              buf.lang = opposite;
              buf.langConfident = false;
              this.logger.log(`🔄 UniversalFlip [${buf.lastEmittedLang}→${opposite}] "${text}" [${sessionId}]`);
            }
          }
        } else if (bufEmpty) {
          if (hasRealLang) {
            const lang = aaiLang.startsWith('es') ? 'es' : 'en';
            buf.lang = lang;
            buf.langConfident = true;
            if (lang !== buf.lastEmittedLang) {
              this.logger.log(`🌍 LangFromAAI [${buf.lastEmittedLang}→${lang}] [${sessionId}]`);
            }
          } else if (lexStrong && buf.lastEmittedLang && buf.lastEmittedLang !== lexLang) {
            buf.lang = lexLang;
            buf.langConfident = true;
            this.logger.log(`🌍 LangFromLex [${buf.lastEmittedLang}→${lexLang}] [${sessionId}]`);
          } else if (!lexStrong && this.isBackchannel(text) && buf.lastEmittedLang) {
            const isSpanishBackchannel = /^(sí|si)\.?,?$/i.test(text.trim());
            if (isSpanishBackchannel && buf.lastEmittedLang === 'es') {
              buf.lang = 'es';
            } else {
              const opposite = buf.lastEmittedLang === 'en' ? 'es' : 'en';
              buf.lang = opposite;
              this.logger.log(`🔄 BackchanelFlip [${buf.lastEmittedLang}→${opposite}] "${text}" [${sessionId}]`);
            }
            buf.langConfident = false;
          } else {
            buf.lang = this.resolveLang(text, aaiLang, aaiConf, null, wordCount);
            buf.langConfident = hasRealLang;
          }
        } else if (hasRealLang) {
          const newLang = aaiLang.startsWith('es') ? 'es' : 'en';
          if (!buf.langConfident && newLang !== buf.lang) {
            this.logger.log(`🌍 LangCorrect [${buf.lang}→${newLang}] conf=${aaiConf.toFixed(2)} [${sessionId}]`);
            buf.lang = newLang;
            buf.langConfident = true;
          } else if (aaiConf > 0.80) {
            buf.lang = newLang;
            buf.langConfident = true;
          } else if (silenceGap && aaiConf > 0.40 && newLang !== buf.lang) {
            this.logger.log(`🌍 LangUpdate [${buf.lang}→${newLang}] gap+conf=${aaiConf.toFixed(2)} [${sessionId}]`);
            buf.lang = newLang;
            buf.langConfident = true;
          }
        }

        if (hasRealLang) {
          this.logger.log(`🌐 ASR ${aaiLang} conf=${aaiConf.toFixed(2)} → ${buf.lang} (lex=${lexLang} strong=${lexStrong}) [${sessionId}]`);
        }

        // ── Speaker change ─────────────────────────────────────────────────────
        const isGrowingTurn = buf.text && text.startsWith(buf.text.trimEnd());
        const detectedLang = this.resolveLang(text, aaiLang, aaiConf, buf.lang, wordCount);
        const confOk = aaiConf >= MIN_SPEAKER_CHANGE_CONF && wordCount >= 2;
        const veryConf = aaiConf >= 0.8;
        const lexConfChange = lexStrong && buf.lang && buf.lang !== lexLang && buf.text;
        const bufLangChanged = buf.lang && buf.lang !== detectedLang && buf.text;

        if (!isGrowingTurn && silenceGap &&
          ((bufLangChanged && (confOk || veryConf)) || (lexConfChange && wordCount >= 3))) {
          this.logger.log(`🔀 SpeakerChange [${buf.lang}→${detectedLang}] gap=${now - buf.lastUpdateMs}ms [${sessionId}]`);
          this.closeTurn(sessionId, 'speakerChange');
          buf.lang = detectedLang;
          buf.langConfident = hasRealLang;
        }

        buf.lastUpdateMs = now;
        buf.text = text;
        this.emitPartial(s, sessionId);
        this.logger.log(`📝 ${isFinal ? 'FINAL' : 'Part'} [${buf.lang}] "${text.substring(0, 80)}" [${sessionId}]`);

        // ── FIX #6: Split de Turn mezclado — pregunta EN + respuesta ES ──────
        // Cuando el doctor confirma algo en inglés y el paciente responde
        // inmediatamente, u3-rt-pro puede fusionarlos:
        // "You have had no seizures after the medication increase? No, doctor."
        // Detectar patrón: texto en inglés que termina en "? <respuesta_es>"
        // y partir en dos bloques: el inglés y la respuesta española.
        if (isFinal && buf.lang === 'en') {
          const mixedMatch = text.match(/^(.+\?)\s+((?:sí|si|no|claro|bien|okay|ok|cuatro|four|tres|three|dos|two|uno|one|\d+)[^?]*)$/i);
          if (mixedMatch) {
            const enPart = mixedMatch[1].trim();
            const esPart = mixedMatch[2].trim();
            const enWords = enPart.split(/\s+/).filter(Boolean);
            // Solo partir si la parte EN tiene ≥4 palabras (es una pregunta real)
            // y la parte ES es claramente una respuesta (≤6 palabras)
            const esWords = esPart.split(/\s+/).filter(Boolean).length;
            const hasEnContent = enWords.some(w => /^(the|you|have|had|are|was|were|do|does|did|your|any|ever|since|before|after|how|when|what|why)$/i.test(w));
            if (enWords.length >= 4 && esWords <= 6 && hasEnContent) {
              this.logger.log(`✂️ Split EN+ES [${sessionId}]: EN="${enPart.substring(0,50)}" ES="${esPart}"`);
              // Emitir parte EN como turno del doctor
              buf.text = enPart;
              buf.peakText = enPart;
              buf.lang = 'en';
              this.clearTimer(buf);
              await this.closeTurn(sessionId, 'splitMixed');
              // Emitir parte ES como turno del paciente
              const sAfterSplit = this.sessionData.get(sessionId);
              if (sAfterSplit) {
                sAfterSplit.buffer.text = esPart;
                sAfterSplit.buffer.peakText = esPart;
                sAfterSplit.buffer.lang = 'es';
                sAfterSplit.buffer.langConfident = false;
                sAfterSplit.buffer.lastUpdateMs = now;
                await this.closeTurn(sessionId, 'splitMixed');
              }
              return;
            }
          }
        }

        // ── ForceClose por mezcla EN+ES ────────────────────────────────────────
        if (wordCount >= 8 && buf.text) {
          const words = text.trim().split(/\s+/);
          const esOnly = /^(que|los|las|del|una|con|para|pero|desde|hace|porque|también|cuando|como|esto|eso|fue|han|tengo|tuve|tenía|convulsiones|días|mes|año|años|siempre|nunca|alguna|dejé|pagar|cobraba|incrementaron|tomarla|todos)$/i;
          const enOnly = /^(the|and|you|have|had|are|taking|medications|seizures|since|before|after|dose|increase|missed|those|pills|times|every|medical|conditions|family|history|examine|when|was|your|last|seizure|not)$/i;
          const lastThird = words.slice(Math.floor(words.length * 0.6));
          const firstHalf = words.slice(0, Math.floor(words.length * 0.5));
          const fEN = firstHalf.some(w => enOnly.test(w));
          const fES = firstHalf.some(w => esOnly.test(w));
          const lEN = lastThird.some(w => enOnly.test(w));
          const lES = lastThird.some(w => esOnly.test(w));
          const mixConf = (fEN ? 1 : 0) + (fES ? 1 : 0) + (lEN ? 1 : 0) + (lES ? 1 : 0);
          if (((fEN && lES) || (fES && lEN)) && mixConf >= 3) {
            this.logger.log(`🔀 ForceClose mezcla [${sessionId}] conf=${mixConf} "${text.substring(0, 60)}"`);
            this.clearTimer(buf);
            buf.forceClosedMs = now;
            this.closeTurn(sessionId, 'silence');
            return;
          }
        }

        // ── Silence timer ──────────────────────────────────────────────────────
        const textGrew = text !== buf.lastSeenText;
        buf.lastSeenText = text;
        if (textGrew) {
          buf.staleCount = 0;
          this.clearTimer(buf);
          buf.timer = setTimeout(() => {
            buf.timer = null;
            this.logger.log(`⏱ Silence close [${sessionId}]`);
            this.closeTurn(sessionId, 'silence');
          }, T_SILENCE_CLOSE);
        } else {
          buf.staleCount++;
          if (buf.staleCount === 3) this.logger.log(`🧊 Turn estancado [${sessionId}] stale=${buf.staleCount}`);
          if (!buf.timer) {
            const staleWords = buf.text.trim().split(/\s+/).filter(Boolean).length;
            // FIX #1: 400ms → 700ms para dar tiempo a u3-rt-pro de terminar
            // la última sílaba antes del ForceEndpoint. Con 400ms el modelo
            // recibía la interrupción antes de cerrar fonemas finales.
            const closeDelay = staleWords > 12 ? 700 : T_SILENCE_CLOSE_STALE;
            buf.timer = setTimeout(() => {
              buf.timer = null;
              const s2 = this.sessionData.get(sessionId);
              if (s2?.ws?.readyState === 1 && s2.buffer.text) {
                this.logger.log(`⚡ ForceEndpoint [${sessionId}] (${staleWords}w)`);
                s2.ws.send(JSON.stringify({ type: 'ForceEndpoint' }));
                const fb = setTimeout(() => {
                  this.logger.log(`⏱ Silence close fallback [${sessionId}]`);
                  this.closeTurn(sessionId, 'silence');
                }, 600);
                const s3 = this.sessionData.get(sessionId);
                if (s3) (s3 as any)._forceEndpointFallback = fb;
              } else {
                this.logger.log(`⏱ Silence close [${sessionId}]`);
                this.closeTurn(sessionId, 'silence');
              }
            }, closeDelay);
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
      if (s.chunkCount % 20 === 0) this.logger.log(`📤 [${sessionId}] Chunk #${s.chunkCount}`);
      if (ws.readyState === WebSocket.OPEN) ws.send(chunk);
    };

    const close = async () => {
      this.logger.log(`⏳ Cerrando AAI v3 [${sessionId}]`);
      const s = this.sessionData.get(sessionId);
      if (s?.buffer.text || s?.buffer.peakText) await this.closeTurn(sessionId, 'userStop');
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'Terminate' }));
        await new Promise((r) => setTimeout(r, 2500));
      }
      ws.close();
      this.logger.log(`🛑 AAI v3 cerrado [${sessionId}]`);
    };

    return { send, close };
  }

  private async claudePipeline(text: string, lang: 'es' | 'en', session: SessionData, sessionId: string) {
    const history = [...session.conversationHistory];
    const { result, correctedLang } = await this.correctWithClaude(text, lang, history);
    if (result !== text || correctedLang !== lang) {
      this.logger.log(`✨ CLAUDE [${lang}→${correctedLang}]: "${result.substring(0, 80)}"`);
      const idx = session.conversationHistory.findLastIndex((t) => t.text === text);
      if (idx >= 0) { session.conversationHistory[idx].text = result; session.conversationHistory[idx].lang = correctedLang; }
      if (correctedLang !== lang && session.buffer.lastEmittedLang === lang) session.buffer.lastEmittedLang = correctedLang;
      this.emit(session, { text: result, language: correctedLang, isCorrection: true, originalText: text, sessionId });
    }
  }

  private async correctWithClaude(
    text: string, lang: 'es' | 'en', history: ConversationTurn[],
  ): Promise<{ result: string; correctedLang: 'es' | 'en' }> {
    if (!this.anthropic || text.length < 5) return { result: text, correctedLang: lang };
    const ctx = history.slice(0, -1).slice(-5)
      .map((t) => `[${t.lang === 'en' ? 'Doctor' : 'Patient'}]: ${t.text}`).join('\n');

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
8. CRITICAL — Wrong language detection: If the detected language is English but the text looks like garbled Spanish, correct it. Example: "See those mean." → "Sí, dos mil."

Output ONLY the corrected text — no explanations, no quotes.`;

    try {
      const r = await this.anthropic.messages.create({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 300,
        messages: [{ role: 'user', content: prompt }],
      });
      const result = (r.content[0] as any).text?.trim() || text;
      if (this.norm(result) === this.norm(text)) return { result: text, correctedLang: lang };
      if (result.length > text.length * 1.4 + 20) return { result: text, correctedLang: lang };
      const correctedLang: 'es' | 'en' = this.detectLang(result) ?? lang;
      return { result, correctedLang };
    } catch (e: any) {
      this.logger.error(`❌ Claude: ${e.message}`);
      return { result: text, correctedLang: lang };
    }
  }
}

// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';
// import Anthropic from '@anthropic-ai/sdk';

// const T_SILENCE_CLOSE = 1200;
// const T_SILENCE_CLOSE_STALE = 600;
// const MIN_SPEAKER_CHANGE_CONF = 0.72;

// interface TurnBuffer {
//   text: string;
//   lang: 'es' | 'en' | null;
//   lastUpdateMs: number;
//   lastClosedMs: number;
//   lastEmittedText: string;
//   lastEmittedLang: 'es' | 'en' | null;
//   timer: NodeJS.Timeout | null;
//   lastSeenText: string;
//   staleCount: number;
//   forceClosedMs: number;
//   peakText: string;
//   langConfident: boolean;
//   // Debounce de partials — timestamp del último partial emitido
//   lastPartialEmitMs: number;
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
//   ws?: any;
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
//       peakText: '',
//       langConfident: false,
//       lastPartialEmitMs: 0,
//     };
//   }

//   private clearTimer(buf: TurnBuffer) {
//     if (buf.timer) { clearTimeout(buf.timer); buf.timer = null; }
//   }

//   private resetBuffer(buf: TurnBuffer) {
//     buf.text = '';
//     buf.lang = null;
//     buf.lastUpdateMs = 0;
//     buf.timer = null;
//     buf.lastSeenText = '';
//     buf.staleCount = 0;
//     buf.peakText = '';
//     buf.langConfident = false;
//     buf.lastPartialEmitMs = 0;
//   }

//   private detectLang(text: string): 'es' | 'en' {
//     const t = text.toLowerCase();
//     const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar)\b/g) || []).length;
//     const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes)\b/g) || []).length;
//     return esScore > enScore ? 'es' : 'en';
//   }

//   private detectLangWithStrength(text: string): { lang: 'es' | 'en'; strong: boolean } {
//     const t = text.toLowerCase();
//     const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar)\b/g) || []).length;
//     const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes)\b/g) || []).length;
//     const lang = esScore > enScore ? 'es' : 'en';
//     const strong = Math.max(esScore, enScore) >= 2 || Math.abs(esScore - enScore) >= 2;
//     return { lang, strong };
//   }

//   private resolveLang(
//     text: string, aaiLang: string | undefined, aaiConf: number,
//     bufLang: 'es' | 'en' | null, wordCount: number,
//   ): 'es' | 'en' {
//     if (aaiLang && aaiConf > 0.40 && aaiLang !== 'undefined') {
//       return aaiLang.startsWith('es') ? 'es' : 'en';
//     }
//     const { lang: lexLang, strong } = this.detectLangWithStrength(text);
//     if (strong) return lexLang;
//     if (wordCount <= 2 && bufLang) return bufLang;
//     return lexLang;
//   }

//   private fixText(text: string, lang: 'es' | 'en'): string {
//     let t = text.trim();
//     t = t.replace(/\b(keprah?|kepra|quepra|kephra|kebri[ah]?|kebra)\b/gi, 'Keppra');
//     if (lang === 'es') t = t.replace(/^(see|si)\s/i, 'Sí, ').replace(/\b2[\s,]?000\b/g, '2,000');
//     if (lang === 'en') t = t.replace(/\b2[\s,]?000\b/g, '2,000');
//     const enStartWords = /^(or|before|after|the|was|were|is|are|have|had|do|does|did|when|where|what|how|why|which|that|this|it|in|of|for|with|a|an|and|but|not|no|any|all|one|two|three|four|some|your|their|our|my|its)/i;
//     const shortPrefixMatch = t.match(/^(\d{1,3}\.?\s+)(\w.+)/);
//     if (shortPrefixMatch && enStartWords.test(shortPrefixMatch[2])) {
//       t = shortPrefixMatch[2].charAt(0).toUpperCase() + shortPrefixMatch[2].slice(1);
//     }
//     const firstWord = t.split(/\s+/)[0]?.replace(/[.,!?¿¡]/g, '').toLowerCase() ?? '';
//     const isCont = /^(pude|pudo|puede|me|te|se|lo|la|le|los|las|y|e|o|pero|que|porque|aunque|cuando|and|or|but|so|because|since|though|however)$/.test(firstWord);
//     if (!isCont && t.length > 0) t = t.charAt(0).toUpperCase() + t.slice(1);
//     return t;
//   }

//   private norm(s: string): string {
//     return s.replace(/[.,;:!?¿¡]/g, '').toLowerCase().replace(/\s+/g, ' ').trim();
//   }

//   private isBackchannel(text: string): boolean {
//     const t = text.trim().replace(/[.!?¿¡,]/g, '').toLowerCase();
//     if (/^\d{1,3}$/.test(t)) return true;
//     return /^(sí|si|no|okay|ok|claro|bueno|bien|ajá|aja|mhm|yes|yeah|nope|cuatro|four|tres|three|dos|two|uno|one)$/.test(t);
//   }

//   private emit(session: SessionData, payload: object) {
//     session.callback(JSON.stringify(payload));
//   }

//   private emitPartial(session: SessionData, sessionId: string) {
//     const buf = session.buffer;
//     if (!buf.text || !buf.lang) return;
//     const words = buf.text.trim().split(/\s+/).filter(Boolean).length;
//     const isKnownBackchannel = /^(sí|si|no|yes|ok|yeah|cuatro|four|tres|three|dos|two|uno|one|bien|claro)\.?,?$/i.test(buf.text.trim());
//     if (words < 2 && !isKnownBackchannel) return;

//     // Debounce: no emitir partials más rápido que cada 150ms
//     // Esto reduce el lag visible en el frontend sin perder información
//     const now = Date.now();
//     if (now - buf.lastPartialEmitMs < 150) return;
//     buf.lastPartialEmitMs = now;

//     this.emit(session, { text: buf.text, language: buf.lang, isNewTurn: false, sessionId });
//   }

//   private async closeTurn(sessionId: string, reason: string): Promise<void> {
//     const session = this.sessionData.get(sessionId);
//     if (!session) return;
//     const buf = session.buffer;
//     const textToClose = (buf.peakText && buf.peakText.length > (buf.text?.length || 0))
//       ? buf.peakText : buf.text;
//     if (!textToClose) return;

//     this.clearTimer(buf);
//     const lang = buf.lang ?? this.detectLang(textToClose);
//     const finalText = this.fixText(textToClose, lang);
//     if (!finalText) { this.resetBuffer(buf); return; }

//     const wordCount = finalText.trim().split(/\s+/).length;
//     const isUniversalBackchannel = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(finalText.trim());
//     const isNumericResponse = /^\d+\.?$/.test(finalText.trim());

//     // Filtro de eco: palabra única que ya aparece al final de un bloque reciente
//     // EXCEPCIÓN: backchannels universales y respuestas numéricas siempre pasan
//     if (wordCount === 1 && !isUniversalBackchannel && !isNumericResponse) {
//       const w = this.norm(finalText);
//       const prev = this.norm(buf.lastEmittedText ?? '');
//       if (prev.endsWith(w)) { this.resetBuffer(buf); return; }
//       const recentHistory = session.conversationHistory.slice(-5);
//       for (const h of recentHistory) {
//         if (this.norm(h.text).endsWith(w)) {
//           this.logger.log(`🔇 Eco [${sessionId}]: "${finalText}"`);
//           this.resetBuffer(buf);
//           return;
//         }
//       }
//     }

//     const isShortBackchannel = wordCount <= 2;
//     if (!isShortBackchannel && this.norm(finalText) === this.norm(buf.lastEmittedText)) {
//       this.logger.log(`⏭ Dedup [${lang}] [${sessionId}]`);
//       this.resetBuffer(buf);
//       return;
//     }

//     // Filtro de ruido de 1 char (ej: "C." que AAI emite como eco)
//     if (finalText.trim().length === 1 && !isUniversalBackchannel && !isNumericResponse) {
//       this.logger.log(`🚫 Ruido 1-char [${sessionId}]: "${finalText}"`);
//       this.resetBuffer(buf);
//       return;
//     }

//     this.logger.log(`✅ CLOSE [${lang}] [${sessionId}] (${reason}): "${finalText.substring(0, 80)}"`);
//     buf.lastEmittedText = finalText;
//     buf.lastEmittedLang = lang;
//     buf.lastClosedMs = Date.now();

//     this.emit(session, { text: finalText, language: lang, isNewTurn: true, isForcedClose: false, sessionId });
//     session.conversationHistory.push({ lang, text: finalText });
//     if (session.conversationHistory.length > 20) session.conversationHistory.shift();

//     this.resetBuffer(buf);
//     this.claudePipeline(finalText, lang, session, sessionId);
//   }

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

//     // u3-rt-pro: modelo recomendado por AssemblyAI.
//     // Detecta fin de turno por puntuación (no por confianza), code-switching
//     // EN/ES nativo sin parámetros extra, sub-300ms. Esta es la causa raíz
//     // de los turnos mezclados — universal-streaming-multilingual acumulaba
//     // audio de doctor+paciente en un solo Turn por su sistema de confianza.
//     const params = new URLSearchParams({
//       sample_rate: '16000',
//       format_turns: 'true',
//       speech_model: 'u3-rt-pro',
//     });

//     // u3-rt-pro acepta prompt para guiar la transcripción y keyterms para
//     // vocabulario médico específico. El prompt indica que es conversación
//     // bilingüe médica EN/ES para mejorar el code-switching.
//     const U3_PROMPT = 'Bilingual medical interpreter conversation. Doctor speaks English, patient speaks Spanish. Medical terminology includes seizures, Keppra, epilepsy, convulsiones, medicamentos.';

//     const KEYTERMS = [
//       'Keppra', 'convulsión', 'convulsiones', 'epilepsia',
//       'seizure', 'seizures', 'levetiracetam', 'medicamento',
//       'medicamentos', 'valproato', 'carbamazepina', 'lamotrigina',
//       'cerebro', 'dosis',
//     ];

//     const WebSocket = require('ws');
//     const ws = new WebSocket(
//       `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`,
//       { headers: { Authorization: apiKey } },
//     );

//     session.ws = ws;
//     ws.on('open', () => this.logger.log(`✅ AssemblyAI v3 abierto [${sessionId}]`));
//     ws.on('error', (err: Error) => this.logger.error(`❌ AAI error [${sessionId}]: ${err.message}`));

//     ws.on('close', (code: number) => {
//       this.logger.log(`🔒 AAI cerrado [${sessionId}] (${code})`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) this.closeTurn(sessionId, 'streamClose');
//       this.sessionData.delete(sessionId);
//     });

//     ws.on('message', (raw: any) => {
//       const s = this.sessionData.get(sessionId);
//       if (!s) return;
//       let msg: any;
//       try { msg = JSON.parse(raw.toString()); } catch { return; }

//       const buf = s.buffer;
//       const now = Date.now();

//       if (msg.type === 'Begin') {
//         this.logger.log(`🔗 AAI v3 [${sessionId}] sid=${msg.id}`);
//         ws.send(JSON.stringify({ type: 'UpdateConfiguration', keyterms: KEYTERMS, prompt: U3_PROMPT }));
//         return;
//       }

//       if (msg.type === 'Turn') {
//         const text: string = (msg.transcript || '').trim();
//         const aaiLang: string = msg.language_code ?? '';
//         const aaiConf: number = msg.language_confidence ?? 0;
//         const isFinal: boolean = msg.turn_is_formatted === true;
//         const wordCount = text.split(/\s+/).filter(Boolean).length;

//         // hasRealLang: AAI envió un idioma concreto con confianza > 0
//         // Ignorar 'undefined' (string) que AAI envía cuando no sabe
//         const hasRealLang = !!(aaiLang && aaiLang !== 'undefined' && aaiConf > 0);

//         // isUniversalWord: palabras que son válidas en cualquier idioma
//         // Para estas, SIEMPRE confiar en el contexto (lastEmittedLang flip)
//         // y NUNCA filtrar por ser idioma no objetivo
//         const isUniversalWord = /^(no|sí|si|yes|ok|yeah|bien|\d+)\.?,?$/i.test(text.trim());

//         if (isFinal && (s as any)._forceEndpointFallback) {
//           clearTimeout((s as any)._forceEndpointFallback);
//           delete (s as any)._forceEndpointFallback;
//         }

//         this.logger.log(`🔬 RAW fmt=${isFinal} lang=${aaiLang} conf=${aaiConf.toFixed(2)} "${text.substring(0, 60)}" [${sessionId}]`);
//         if (!text) return;

//         // ── Filtro de ruido ────────────────────────────────────────────────────
//         // Descartar idiomas no objetivo con baja confianza y texto corto
//         // PERO nunca descartar palabras universales (no/si/sí/yes/números)
//         const isNonTargetLang = hasRealLang && aaiLang !== 'en' && aaiLang !== 'es';
//         if (isNonTargetLang && aaiConf < 0.65 && wordCount <= 2 && !isUniversalWord) {
//           this.logger.log(`🚫 Ruido [${aaiLang}=${aaiConf.toFixed(2)}] "${text}" [${sessionId}]`);
//           return;
//         }

//         // Actualizar peakText
//         if (text.length > (buf.peakText?.length || 0)) buf.peakText = text;

//         // ── Guard de continuación post-close ──────────────────────────────────
//         const msSinceClose = now - buf.lastClosedMs;
//         const msSinceForceClose = now - buf.forceClosedMs;
//         if (!buf.text && msSinceClose < 1200 && buf.lastEmittedText && msSinceForceClose >= 2000) {
//           const normalize = (str: string) =>
//             str.replace(/Keppra/gi, 'kepra').replace(/[,\.!?¿¡]/g, '').replace(/\s+/g, ' ').trim().toLowerCase();
//           const prefix = normalize(buf.lastEmittedText).substring(0, 20);
//           if (prefix.length >= 4 && normalize(text).startsWith(prefix)) {
//             this.logger.log(`🔁 ContinuationGuard [${sessionId}] +${msSinceClose}ms`);
//             buf.text = text;
//             buf.peakText = text;
//             buf.lang = buf.lastEmittedLang;
//             buf.langConfident = true;
//             this.clearTimer(buf);
//             buf.timer = setTimeout(() => { buf.timer = null; this.closeTurn(sessionId, 'silence'); }, T_SILENCE_CLOSE);
//             return;
//           }
//         }

//         const { lang: lexLang, strong: lexStrong } = this.detectLangWithStrength(text);
//         const silenceGap = now - buf.lastUpdateMs > 400;
//         const bufEmpty = !buf.lang || !buf.text;

//         // ── Asignación de idioma ───────────────────────────────────────────────
//         // Lógica unificada con prioridades claras:
//         //
//         // CASO ESPECIAL: palabras universales (no/si/sí/yes/números)
//         //   → Siempre hacer BackchanelFlip basado en lastEmittedLang
//         //   → No importa lo que diga AAI (puede decir 'it', 'fr', etc.)
//         //
//         // CASO NORMAL:
//         //   1. Buffer vacío + AAI con idioma real → LangFromAAI
//         //   2. Buffer vacío + léxico fuerte diferente → LangFromLex
//         //   3. Buffer vacío + backchannel ambiguo → BackchanelFlip
//         //   4. Buffer con texto + lang no confident + AAI real → LangCorrect
//         //   5. Buffer con texto + conf alta (>0.8) → actualizar siempre
//         //   6. Buffer con texto + conf media + silencio → LangUpdate

//         if (isUniversalWord && buf.lastEmittedLang) {
//           // Para palabras universales: flip siempre al idioma contrario
//           // EXCEPCIÓN: si/sí/no después de español → mantener español
//           const isSpanishResponse = /^(sí|si|no)\.?,?$/i.test(text.trim());
//           if (isSpanishResponse && buf.lastEmittedLang === 'es') {
//             buf.lang = 'es';
//             buf.langConfident = false;
//           } else {
//             const opposite = buf.lastEmittedLang === 'en' ? 'es' : 'en';
//             buf.lang = opposite;
//             buf.langConfident = false;
//             this.logger.log(`🔄 UniversalFlip [${buf.lastEmittedLang}→${opposite}] "${text}" [${sessionId}]`);
//           }
//         } else if (bufEmpty) {
//           if (hasRealLang) {
//             const lang = aaiLang.startsWith('es') ? 'es' : 'en';
//             buf.lang = lang;
//             buf.langConfident = true;
//             if (lang !== buf.lastEmittedLang) {
//               this.logger.log(`🌍 LangFromAAI [${buf.lastEmittedLang}→${lang}] [${sessionId}]`);
//             }
//           } else if (lexStrong && buf.lastEmittedLang && buf.lastEmittedLang !== lexLang) {
//             buf.lang = lexLang;
//             buf.langConfident = true;
//             this.logger.log(`🌍 LangFromLex [${buf.lastEmittedLang}→${lexLang}] [${sessionId}]`);
//           } else if (!lexStrong && this.isBackchannel(text) && buf.lastEmittedLang) {
//             const isSpanishBackchannel = /^(sí|si|no)\.?,?$/i.test(text.trim());
//             if (isSpanishBackchannel && buf.lastEmittedLang === 'es') {
//               buf.lang = 'es';
//             } else {
//               const opposite = buf.lastEmittedLang === 'en' ? 'es' : 'en';
//               buf.lang = opposite;
//               this.logger.log(`🔄 BackchanelFlip [${buf.lastEmittedLang}→${opposite}] "${text}" [${sessionId}]`);
//             }
//             buf.langConfident = false;
//           } else {
//             buf.lang = this.resolveLang(text, aaiLang, aaiConf, null, wordCount);
//             buf.langConfident = hasRealLang;
//           }
//         } else if (hasRealLang) {
//           const newLang = aaiLang.startsWith('es') ? 'es' : 'en';
//           if (!buf.langConfident && newLang !== buf.lang) {
//             this.logger.log(`🌍 LangCorrect [${buf.lang}→${newLang}] conf=${aaiConf.toFixed(2)} [${sessionId}]`);
//             buf.lang = newLang;
//             buf.langConfident = true;
//           } else if (aaiConf > 0.80) {
//             buf.lang = newLang;
//             buf.langConfident = true;
//           } else if (silenceGap && aaiConf > 0.40 && newLang !== buf.lang) {
//             this.logger.log(`🌍 LangUpdate [${buf.lang}→${newLang}] gap+conf=${aaiConf.toFixed(2)} [${sessionId}]`);
//             buf.lang = newLang;
//             buf.langConfident = true;
//           }
//         }

//         if (hasRealLang) {
//           this.logger.log(`🌐 ASR ${aaiLang} conf=${aaiConf.toFixed(2)} → ${buf.lang} (lex=${lexLang} strong=${lexStrong}) [${sessionId}]`);
//         }

//         // ── Speaker change ─────────────────────────────────────────────────────
//         const isGrowingTurn = buf.text && text.startsWith(buf.text.trimEnd());
//         const detectedLang = this.resolveLang(text, aaiLang, aaiConf, buf.lang, wordCount);
//         const confOk = aaiConf >= MIN_SPEAKER_CHANGE_CONF && wordCount >= 2;
//         const veryConf = aaiConf >= 0.8;
//         const lexConfChange = lexStrong && buf.lang && buf.lang !== lexLang && buf.text;
//         const bufLangChanged = buf.lang && buf.lang !== detectedLang && buf.text;

//         if (!isGrowingTurn && silenceGap &&
//           ((bufLangChanged && (confOk || veryConf)) || (lexConfChange && wordCount >= 3))) {
//           this.logger.log(`🔀 SpeakerChange [${buf.lang}→${detectedLang}] gap=${now - buf.lastUpdateMs}ms [${sessionId}]`);
//           this.closeTurn(sessionId, 'speakerChange');
//           buf.lang = detectedLang;
//           buf.langConfident = hasRealLang;
//         }

//         buf.lastUpdateMs = now;
//         buf.text = text;
//         this.emitPartial(s, sessionId);
//         this.logger.log(`📝 ${isFinal ? 'FINAL' : 'Part'} [${buf.lang}] "${text.substring(0, 80)}" [${sessionId}]`);

//         // ── ForceClose por mezcla EN+ES ────────────────────────────────────────
//         if (wordCount >= 8 && buf.text) {
//           const words = text.trim().split(/\s+/);
//           const esOnly = /^(que|los|las|del|una|con|para|pero|desde|hace|porque|también|cuando|como|esto|eso|fue|han|tengo|tuve|tenía|convulsiones|días|mes|año|años|siempre|nunca|alguna|dejé|pagar|cobraba|incrementaron|tomarla|todos)$/i;
//           const enOnly = /^(the|and|you|have|had|are|taking|medications|seizures|since|before|after|dose|increase|missed|those|pills|times|every|medical|conditions|family|history|examine|when|was|your|last|seizure|not)$/i;
//           const lastThird = words.slice(Math.floor(words.length * 0.6));
//           const firstHalf = words.slice(0, Math.floor(words.length * 0.5));
//           const fEN = firstHalf.some(w => enOnly.test(w));
//           const fES = firstHalf.some(w => esOnly.test(w));
//           const lEN = lastThird.some(w => enOnly.test(w));
//           const lES = lastThird.some(w => esOnly.test(w));
//           const mixConf = (fEN ? 1 : 0) + (fES ? 1 : 0) + (lEN ? 1 : 0) + (lES ? 1 : 0);
//           if (((fEN && lES) || (fES && lEN)) && mixConf >= 3) {
//             this.logger.log(`🔀 ForceClose mezcla [${sessionId}] conf=${mixConf} "${text.substring(0, 60)}"`);
//             this.clearTimer(buf);
//             buf.forceClosedMs = now;
//             this.closeTurn(sessionId, 'silence');
//             return;
//           }
//         }

//         // ── Silence timer ──────────────────────────────────────────────────────
//         const textGrew = text !== buf.lastSeenText;
//         buf.lastSeenText = text;
//         if (textGrew) {
//           buf.staleCount = 0;
//           this.clearTimer(buf);
//           buf.timer = setTimeout(() => {
//             buf.timer = null;
//             this.logger.log(`⏱ Silence close [${sessionId}]`);
//             this.closeTurn(sessionId, 'silence');
//           }, T_SILENCE_CLOSE);
//         } else {
//           buf.staleCount++;
//           if (buf.staleCount === 3) this.logger.log(`🧊 Turn estancado [${sessionId}] stale=${buf.staleCount}`);
//           if (!buf.timer) {
//             const staleWords = buf.text.trim().split(/\s+/).filter(Boolean).length;
//             const closeDelay = staleWords > 12 ? 400 : T_SILENCE_CLOSE_STALE;
//             buf.timer = setTimeout(() => {
//               buf.timer = null;
//               const s2 = this.sessionData.get(sessionId);
//               if (s2?.ws?.readyState === 1 && s2.buffer.text) {
//                 this.logger.log(`⚡ ForceEndpoint [${sessionId}] (${staleWords}w)`);
//                 s2.ws.send(JSON.stringify({ type: 'ForceEndpoint' }));
//                 const fb = setTimeout(() => {
//                   this.logger.log(`⏱ Silence close fallback [${sessionId}]`);
//                   this.closeTurn(sessionId, 'silence');
//                 }, 600);
//                 const s3 = this.sessionData.get(sessionId);
//                 if (s3) (s3 as any)._forceEndpointFallback = fb;
//               } else {
//                 this.logger.log(`⏱ Silence close [${sessionId}]`);
//                 this.closeTurn(sessionId, 'silence');
//               }
//             }, closeDelay);
//           }
//         }
//       } else if (msg.type === 'Termination') {
//         this.logger.log(`🏁 Terminado [${sessionId}] audio=${msg.audio_duration_seconds}s`);
//       }
//     });

//     const send = (chunk: ArrayBuffer) => {
//       const s = this.sessionData.get(sessionId);
//       if (!s) return;
//       s.chunkCount++;
//       if (s.chunkCount % 20 === 0) this.logger.log(`📤 [${sessionId}] Chunk #${s.chunkCount}`);
//       if (ws.readyState === WebSocket.OPEN) ws.send(chunk);
//     };

//     const close = async () => {
//       this.logger.log(`⏳ Cerrando AAI v3 [${sessionId}]`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) await this.closeTurn(sessionId, 'userStop');
//       if (ws.readyState === WebSocket.OPEN) {
//         ws.send(JSON.stringify({ type: 'Terminate' }));
//         await new Promise((r) => setTimeout(r, 2500));
//       }
//       ws.close();
//       this.logger.log(`🛑 AAI v3 cerrado [${sessionId}]`);
//     };

//     return { send, close };
//   }

//   private async claudePipeline(text: string, lang: 'es' | 'en', session: SessionData, sessionId: string) {
//     const history = [...session.conversationHistory];
//     const { result, correctedLang } = await this.correctWithClaude(text, lang, history);
//     if (result !== text || correctedLang !== lang) {
//       this.logger.log(`✨ CLAUDE [${lang}→${correctedLang}]: "${result.substring(0, 80)}"`);
//       const idx = session.conversationHistory.findLastIndex((t) => t.text === text);
//       if (idx >= 0) { session.conversationHistory[idx].text = result; session.conversationHistory[idx].lang = correctedLang; }
//       if (correctedLang !== lang && session.buffer.lastEmittedLang === lang) session.buffer.lastEmittedLang = correctedLang;
//       this.emit(session, { text: result, language: correctedLang, isCorrection: true, originalText: text, sessionId });
//     }
//   }

//   private async correctWithClaude(
//     text: string, lang: 'es' | 'en', history: ConversationTurn[],
//   ): Promise<{ result: string; correctedLang: 'es' | 'en' }> {
//     if (!this.anthropic || text.length < 5) return { result: text, correctedLang: lang };
//     const ctx = history.slice(0, -1).slice(-5)
//       .map((t) => `[${t.lang === 'en' ? 'Doctor' : 'Patient'}]: ${t.text}`).join('\n');

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
// 8. CRITICAL — Wrong language detection: If the detected language is English but the text looks like garbled Spanish, correct it. Example: "See those mean." → "Sí, dos mil."

// Output ONLY the corrected text — no explanations, no quotes.`;

//     try {
//       const r = await this.anthropic.messages.create({
//         model: 'claude-haiku-4-5-20251001',
//         max_tokens: 300,
//         messages: [{ role: 'user', content: prompt }],
//       });
//       const result = (r.content[0] as any).text?.trim() || text;
//       if (this.norm(result) === this.norm(text)) return { result: text, correctedLang: lang };
//       if (result.length > text.length * 1.4 + 20) return { result: text, correctedLang: lang };
//       const correctedLang: 'es' | 'en' = this.detectLang(result) ?? lang;
//       return { result, correctedLang };
//     } catch (e: any) {
//       this.logger.error(`❌ Claude: ${e.message}`);
//       return { result: text, correctedLang: lang };
//     }
//   }
// }
// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';
// import Anthropic from '@anthropic-ai/sdk';

// // ─── Timing ──────────────────────────────────────────────────────────────────
// const T_SILENCE_CLOSE = 1200;
// const T_SILENCE_CLOSE_STALE = 600;
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
//   lastSeenText: string;
//   staleCount: number;
//   forceClosedMs: number;
//   // FIX #7: Rastrea el texto más largo visto en el Turn actual.
//   // Con lang=undefined/conf=0, AAI a veces regresa a un texto más corto
//   // mientras sigue procesando. Guardamos el máximo para no perder palabras.
//   peakText: string;
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
//   ws?: any;
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
//       peakText: '',
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
//     buf.peakText = '';
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
//     if (aaiLang && aaiConf > 0.40)
//       return aaiLang.startsWith('es') ? 'es' : 'en';
//     const { lang: lexLang, strong } = this.detectLangWithStrength(text);
//     if (strong) return lexLang;
//     if (wordCount <= 2 && bufLang) return bufLang;
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

//     const enStartWords =
//       /^(or|before|after|the|was|were|is|are|have|had|do|does|did|when|where|what|how|why|which|that|this|it|in|of|for|with|a|an|and|but|not|no|any|all|one|two|three|four|some|your|their|our|my|its)/i;
//     const shortPrefixMatch = t.match(/^(\d{1,3}\.?\s+)(\w.+)/);
//     if (shortPrefixMatch && enStartWords.test(shortPrefixMatch[2])) {
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
//     const words = buf.text.trim().split(/\s+/).filter(Boolean).length;
//     const isKnownBackchannel = /^(sí|si|no|yes|ok|yeah|cuatro|four|tres|three|dos|two|uno|one|bien|claro)\.?,?$/i.test(buf.text.trim());
//     if (words < 2 && !isKnownBackchannel) return;
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

//     // FIX #7: Usar peakText si buf.text quedó vacío o más corto por regresión de AAI.
//     // Cuando AAI tiene lang=undefined y congela el texto, a veces lo devuelve
//     // más corto que el máximo visto. peakText guarda el texto más largo del Turn.
//     const textToClose = (buf.peakText && buf.peakText.length > (buf.text?.length || 0))
//       ? buf.peakText
//       : buf.text;
//     if (!textToClose) return;

//     this.clearTimer(buf);
//     const lang = buf.lang ?? this.detectLang(textToClose);
//     const finalText = this.fixText(textToClose, lang);
//     if (!finalText) {
//       this.resetBuffer(buf);
//       return;
//     }

//     const wordCount = finalText.trim().split(/\s+/).length;
//     const isUniversalBackchannel = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(
//       finalText.trim(),
//     );
//     if (wordCount === 1 && !isUniversalBackchannel) {
//       const w = this.norm(finalText);
//       const prev = this.norm(buf.lastEmittedText ?? '');
//       if (prev.endsWith(w)) {
//         this.logger.log(
//           `🔇 Eco descartado [${lang}] [${sessionId}]: "${finalText}"`,
//         );
//         this.resetBuffer(buf);
//         return;
//       }
//       const recentHistory = session.conversationHistory.slice(-5);
//       for (const h of recentHistory) {
//         if (this.norm(h.text).endsWith(w)) {
//           this.logger.log(
//             `🔇 Eco descartado (hist) [${lang}] [${sessionId}]: "${finalText}"`,
//           );
//           this.resetBuffer(buf);
//           return;
//         }
//       }
//     }

//     const isShortBackchannel = wordCount <= 2;
//     if (
//       !isShortBackchannel &&
//       this.norm(finalText) === this.norm(buf.lastEmittedText)
//     ) {
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
//       end_of_turn_confidence_threshold: '0.4',
//       max_turn_silence: '600',
//     });

//     const KEYTERMS = [
//       'Keppra', 'convulsión', 'convulsiones', 'epilepsia',
//       'seizure', 'seizures', 'levetiracetam', 'medicamento',
//       'medicamentos', 'valproato', 'carbamazepina', 'lamotrigina',
//       'cerebro', 'dosis',
//     ];

//     const WebSocket = require('ws');
//     const ws = new WebSocket(
//       `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`,
//       { headers: { Authorization: apiKey } },
//     );

//     session.ws = ws;
//     ws.on('open', () =>
//       this.logger.log(`✅ AssemblyAI v3 abierto [${sessionId}]`),
//     );
//     ws.on('error', (err: Error) =>
//       this.logger.error(`❌ AssemblyAI v3 error [${sessionId}]: ${err.message}`),
//     );

//     ws.on('close', (code: number) => {
//       this.logger.log(`🔒 AssemblyAI v3 cerrado [${sessionId}] (${code})`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) this.closeTurn(sessionId, 'streamClose');
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
//         ws.send(JSON.stringify({ type: 'UpdateConfiguration', keyterms: KEYTERMS }));
//         this.logger.log(`📚 Keyterms enviados [${sessionId}]`);
//         return;
//       }

//       if (msg.type === 'Turn') {
//         const text: string = (msg.transcript || '').trim();
//         const aaiLang: string = msg.language_code;
//         const aaiConf: number = msg.language_confidence ?? 0;
//         const isFinal: boolean = msg.turn_is_formatted === true;
//         const wordCount = text.split(/\s+/).filter(Boolean).length;

//         if (isFinal && (s as any)._forceEndpointFallback) {
//           clearTimeout((s as any)._forceEndpointFallback);
//           delete (s as any)._forceEndpointFallback;
//           this.logger.log(`✅ Turn formateado recibido post-ForceEndpoint [${sessionId}]`);
//         }

//         this.logger.log(
//           `🔬 RAW [${sessionId}] fmt=${isFinal} lang=${aaiLang} conf=${aaiConf.toFixed(2)} text="${text.substring(0, 60)}"`,
//         );
//         if (!text) return;

//         // Filtro de ruido
//         const isNonTargetLang =
//           aaiLang && aaiLang !== 'en' && aaiLang !== 'es' && aaiLang !== 'undefined';
//         const isUniversalWord = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(text.trim());
//         if (isNonTargetLang && aaiConf < 0.65 && wordCount <= 2 && !isUniversalWord) {
//           this.logger.log(
//             `🚫 Ruido descartado [${aaiLang}=${aaiConf.toFixed(2)}] "${text}" [${sessionId}]`,
//           );
//           return;
//         }

//         // FIX #7: Actualizar peakText si el texto actual es más largo.
//         // AAI con lang=undefined puede enviar el texto completo en algunos ciclos
//         // y luego regresar a uno más corto. Guardamos el máximo.
//         if (text.length > (buf.peakText?.length || 0)) {
//           buf.peakText = text;
//         }

//         // Guard de continuación post-close
//         const msSinceClose = now - buf.lastClosedMs;
//         const msSinceForceClose = now - buf.forceClosedMs;
//         const forceCloseBlackout = msSinceForceClose < 2000;
//         if (!buf.text && msSinceClose < 1200 && buf.lastEmittedText && !forceCloseBlackout) {
//           const normalize = (s: string) =>
//             s.replace(/Keppra/gi, 'kepra').replace(/[,\.!?¿¡]/g, '').replace(/\s+/g, ' ').trim().toLowerCase();
//           const prevNorm = normalize(buf.lastEmittedText);
//           const curNorm = normalize(text);
//           const prefix = prevNorm.substring(0, Math.min(prevNorm.length, 20));
//           if (prefix.length >= 4 && curNorm.startsWith(prefix)) {
//             this.logger.log(`🔁 ContinuationGuard reopen [${sessionId}] +${msSinceClose}ms`);
//             buf.text = text;
//             buf.peakText = text;
//             buf.lang = buf.lastEmittedLang;
//             this.clearTimer(buf);
//             buf.timer = setTimeout(() => {
//               buf.timer = null;
//               this.closeTurn(sessionId, 'silence');
//             }, T_SILENCE_CLOSE);
//             return;
//           }
//         }

//         const { lang: lexLang, strong: lexStrong } = this.detectLangWithStrength(text);
//         const detectedLang = this.resolveLang(text, aaiLang, aaiConf, buf.lang, wordCount);
//         if (aaiLang)
//           this.logger.log(
//             `🌐 ASR lang=${aaiLang} conf=${aaiConf.toFixed(3)} words=${wordCount} → ${detectedLang} (lex=${lexLang} strong=${lexStrong})`,
//           );

//         // ─── FIX #8: Resolver idioma para frases cortas españolas post-turno EN ──
//         // "Eso fue antes.", "Cuatro." etc. tienen lexStrong=false y AAI da conf baja.
//         // Con lastEmittedLang=en y texto claramente español (aunque léxico débil),
//         // usar el idioma que AAI reportó si es distinto al del buffer anterior.
//         // Antes se quedaba en 'en' porque lexStrong=false y bufLang era null post-close.
//         const bufEmpty = !buf.lang || !buf.text;
//         if (bufEmpty) {
//           // Caso A: AAI dice español con cualquier confianza > 0.40 → confiar
//           if (aaiLang && aaiConf > 0.40) {
//             buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
//             if (buf.lang !== buf.lastEmittedLang) {
//               this.logger.log(
//                 `🌍 LangFromAAI [${buf.lastEmittedLang}→${buf.lang}] post-close [${sessionId}]`,
//               );
//             }
//           } else if (lexStrong && buf.lastEmittedLang && buf.lastEmittedLang !== lexLang) {
//             // Caso B: léxico fuerte señala idioma diferente al turno previo
//             buf.lang = lexLang;
//             this.logger.log(
//               `🌍 LangFromLex [${buf.lastEmittedLang}→${lexLang}] post-close [${sessionId}]`,
//             );
//           } else if (!lexStrong && this.isBackchannel(text) && buf.lastEmittedLang) {
//             // Caso C: backchannel ambiguo → flip de idioma
//             const isSpanishBackchannel = /^(sí|si|no)\.?,?$/i.test(text.trim());
//             if (isSpanishBackchannel && buf.lastEmittedLang === 'es') {
//               buf.lang = 'es';
//               this.logger.log(`🔄 BackchanelKeep [es] text="${text}" [${sessionId}]`);
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

//         // Speaker change
//         const isGrowingTurn = buf.text && text.startsWith(buf.text.trimEnd());
//         const silenceGap = now - buf.lastUpdateMs > 400;
//         const confOk = aaiConf >= MIN_SPEAKER_CHANGE_CONF && wordCount >= 2;
//         const veryConf = aaiConf >= 0.8;
//         const lexConfChange = lexStrong && buf.lang && buf.lang !== lexLang && buf.text;
//         const bufLangChanged = buf.lang && buf.lang !== detectedLang && buf.text;

//         if (
//           !isGrowingTurn && silenceGap &&
//           ((bufLangChanged && (confOk || veryConf)) || (lexConfChange && wordCount >= 3))
//         ) {
//           this.logger.log(
//             `🔀 SpeakerChange [${buf.lang}→${detectedLang}] gap=${now - buf.lastUpdateMs}ms [${sessionId}]`,
//           );
//           this.closeTurn(sessionId, 'speakerChange');
//           buf.lang = detectedLang;
//         }

//         buf.lastUpdateMs = now;
//         buf.text = text;
//         this.emitPartial(s, sessionId);
//         this.logger.log(
//           `📝 ${isFinal ? 'FINAL' : 'Partial'} [${buf.lang}] [${sessionId}]: "${text.substring(0, 80)}"`,
//         );

//         // ForceClose por mezcla de idiomas — requiere 3 señales
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
//           const mixDetected = (firstHasEN && lastHasES) || (firstHasES && lastHasEN);
//           const mixConfidence =
//             (firstHasEN ? 1 : 0) + (firstHasES ? 1 : 0) +
//             (lastHasEN ? 1 : 0) + (lastHasES ? 1 : 0);

//           if (mixDetected && mixConfidence >= 3) {
//             this.logger.log(
//               `🔀 ForceClose por mezcla EN+ES [${sessionId}] conf=${mixConfidence} "${text.substring(0, 60)}"`,
//             );
//             this.clearTimer(buf);
//             buf.forceClosedMs = now;
//             this.closeTurn(sessionId, 'silence');
//             return;
//           }
//         }

//         // Silence timer
//         const textGrew = text !== buf.lastSeenText;
//         buf.lastSeenText = text;
//         if (textGrew) {
//           buf.staleCount = 0;
//           this.clearTimer(buf);
//           buf.timer = setTimeout(() => {
//             buf.timer = null;
//             this.logger.log(`⏱ Silence close [${sessionId}]`);
//             this.closeTurn(sessionId, 'silence');
//           }, T_SILENCE_CLOSE);
//         } else {
//           buf.staleCount++;
//           if (buf.staleCount === 3) {
//             this.logger.log(
//               `🧊 Turn estancado [${sessionId}] stale=${buf.staleCount} — timer no reseteado`,
//             );
//           }
//           if (!buf.timer) {
//             const staleWords = buf.text.trim().split(/\s+/).filter(Boolean).length;
//             const closeDelay = staleWords > 12 ? 400 : T_SILENCE_CLOSE_STALE;
//             buf.timer = setTimeout(() => {
//               buf.timer = null;
//               const s2 = this.sessionData.get(sessionId);
//               if (s2?.ws?.readyState === 1 && s2.buffer.text) {
//                 this.logger.log(`⚡ ForceEndpoint [${sessionId}] (stale turn, ${staleWords}w)`);
//                 s2.ws.send(JSON.stringify({ type: 'ForceEndpoint' }));
//                 const fallbackTimer = setTimeout(() => {
//                   this.logger.log(`⏱ Silence close [${sessionId}] (post-ForceEndpoint fallback)`);
//                   this.closeTurn(sessionId, 'silence');
//                 }, 600);
//                 const s3 = this.sessionData.get(sessionId);
//                 if (s3) (s3 as any)._forceEndpointFallback = fallbackTimer;
//               } else {
//                 this.logger.log(`⏱ Silence close [${sessionId}]`);
//                 this.closeTurn(sessionId, 'silence');
//               }
//             }, closeDelay);
//           }
//         }
//       } else if (msg.type === 'Termination') {
//         this.logger.log(`🏁 Terminado [${sessionId}] audio=${msg.audio_duration_seconds}s`);
//       }
//     });

//     const send = (chunk: ArrayBuffer) => {
//       const s = this.sessionData.get(sessionId);
//       if (!s) return;
//       s.chunkCount++;
//       if (s.chunkCount % 20 === 0)
//         this.logger.log(`📤 [${sessionId}] Chunk #${s.chunkCount}`);
//       if (ws.readyState === WebSocket.OPEN) ws.send(chunk);
//     };

//     const close = async () => {
//       this.logger.log(`⏳ Cerrando AssemblyAI v3 [${sessionId}]`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) await this.closeTurn(sessionId, 'userStop');
//       if (ws.readyState === WebSocket.OPEN) {
//         ws.send(JSON.stringify({ type: 'Terminate' }));
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
//     const { result, correctedLang } = await this.correctWithClaude(text, lang, history);
//     if (result !== text || correctedLang !== lang) {
//       this.logger.log(`✨ CLAUDE [${lang}→${correctedLang}]: "${result.substring(0, 80)}"`);
//       const idx = session.conversationHistory.findLastIndex((t) => t.text === text);
//       if (idx >= 0) {
//         session.conversationHistory[idx].text = result;
//         session.conversationHistory[idx].lang = correctedLang;
//       }
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
//       const detectedResultLang = this.detectLang(result);
//       const correctedLang: 'es' | 'en' = detectedResultLang ?? lang;
//       return { result, correctedLang };
//     } catch (e: any) {
//       this.logger.error(`❌ Claude correct: ${e.message}`);
//       return { result: text, correctedLang: lang };
//     }
//   }
// }
// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';
// import Anthropic from '@anthropic-ai/sdk';

// // ─── Timing ──────────────────────────────────────────────────────────────────
// const T_SILENCE_CLOSE = 800; // 800ms: sincronizado con max_turn_silence=600ms de AAI
// // Reducido de 1200ms — con threshold 0.4 AAI cierra más rápido
// // y necesitamos responder igual de rápido para no perder el siguiente hablante.
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
//   ws?: any; // WebSocket ref para ForceEndpoint
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

//     // ── Filtro de eco: 1 sola palabra que coincide con final de bloques recientes ──
//     // "Increase." aparece como eco del "medication increase." aunque no sea el bloque inmediato anterior.
//     // Buscar en los últimos 5 bloques del historial SOLO para palabras en inglés.
//     // EXCEPCIÓN CRÍTICA: "No.", "Sí.", "Si." son respuestas legítimas del paciente que
//     // se repiten varias veces seguidas — NO deben filtrarse por historial.
//     const wordCount = finalText.trim().split(/\s+/).length;
//     const isUniversalBackchannel = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(
//       finalText.trim(),
//     );
//     if (wordCount === 1 && !isUniversalBackchannel) {
//       const w = this.norm(finalText);
//       // Verificar bloque inmediato anterior
//       const prev = this.norm(buf.lastEmittedText ?? '');
//       if (prev.endsWith(w)) {
//         this.logger.log(
//           `🔇 Eco descartado [${lang}] [${sessionId}]: "${finalText}"`,
//         );
//         this.resetBuffer(buf);
//         return;
//       }
//       // Verificar los últimos 5 bloques del historial (solo para ecos reales, no backchannels)
//       const recentHistory = session.conversationHistory.slice(-5);
//       for (const h of recentHistory) {
//         if (this.norm(h.text).endsWith(w)) {
//           this.logger.log(
//             `🔇 Eco descartado (hist) [${lang}] [${sessionId}]: "${finalText}"`,
//           );
//           this.resetBuffer(buf);
//           return;
//         }
//       }
//     }

//     // ── Dedup: no emitir si es idéntico al bloque anterior ─────────────────────
//     // EXCEPCIÓN: backchannels cortos (≤2 palabras) siempre se emiten aunque
//     // sean iguales — el paciente puede decir "No." / "Sí." varias veces seguidas.
//     const isShortBackchannel = wordCount <= 2;
//     if (
//       !isShortBackchannel &&
//       this.norm(finalText) === this.norm(buf.lastEmittedText)
//     ) {
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
//       end_of_turn_confidence_threshold: '0.4',
//       max_turn_silence: '600',
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

//     session.ws = ws;
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

//         // Si llegó Turn formateado y teníamos un fallback de ForceEndpoint, cancelarlo
//         if (isFinal && (s as any)._forceEndpointFallback) {
//           clearTimeout((s as any)._forceEndpointFallback);
//           delete (s as any)._forceEndpointFallback;
//           this.logger.log(
//             `✅ Turn formateado recibido post-ForceEndpoint [${sessionId}]`,
//           );
//         }

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
//             // Cierre anticipado si hay muchas palabras acumuladas:
//             // Con >12 palabras estancadas el hablante claramente pausó — no esperar
//             // el T_SILENCE_CLOSE completo, usar solo 400ms para liberar el pipeline
//             // antes de que el siguiente hablante empiece.
//             const staleWords = buf.text
//               .trim()
//               .split(/\s+/)
//               .filter(Boolean).length;
//             const closeDelay = staleWords > 12 ? 400 : T_SILENCE_CLOSE;
//             buf.timer = setTimeout(() => {
//               buf.timer = null;
//               // Cuando el Turn está estancado, enviar ForceEndpoint a AAI para que
//               // emita el Turn formateado final con el texto completo correcto.
//               // Luego esperar hasta 600ms para recibir ese Turn antes de cerrar nosotros.
//               const s2 = this.sessionData.get(sessionId);
//               if (s2?.ws?.readyState === 1 /* OPEN */ && s2.buffer.text) {
//                 this.logger.log(
//                   `⚡ ForceEndpoint [${sessionId}] (stale turn, ${staleWords}w)`,
//                 );
//                 s2.ws.send(JSON.stringify({ type: 'ForceEndpoint' }));
//                 // Dar hasta 600ms para que AAI emita el Turn formateado final
//                 const fallbackTimer = setTimeout(() => {
//                   this.logger.log(
//                     `⏱ Silence close [${sessionId}] (post-ForceEndpoint fallback)`,
//                   );
//                   this.closeTurn(sessionId, 'silence');
//                 }, 600);
//                 const s3 = this.sessionData.get(sessionId);
//                 if (s3) (s3 as any)._forceEndpointFallback = fallbackTimer;
//               } else {
//                 this.logger.log(`⏱ Silence close [${sessionId}]`);
//                 this.closeTurn(sessionId, 'silence');
//               }
//             }, closeDelay);
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