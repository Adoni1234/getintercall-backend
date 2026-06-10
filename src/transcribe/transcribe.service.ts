import { Injectable, Logger } from '@nestjs/common';
import { AssemblyAI } from 'assemblyai';
import Anthropic from '@anthropic-ai/sdk';

const T_SILENCE_CLOSE = 800;         // 0.8s: fallback solo si AAI no cierra solo
const T_SILENCE_CLOSE_STALE = 500;  // 0.5s: texto estancado → cerrar rápido
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
    const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar|manejar|dolor|espalda|pregunta|exámenes|resultados|familia|ninguno)\b/g) || []).length;
    const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes|examination|follow|straight|ahead|strength|walking)\b/g) || []).length;
    return esScore > enScore ? 'es' : 'en';
  }

  private detectLangWithStrength(text: string): { lang: 'es' | 'en'; strong: boolean } {
    const t = text.toLowerCase();
    const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar|manejar|dolor|espalda|pregunta|exámenes|resultados|familia|ninguno)\b/g) || []).length;
    const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes|examination|follow|straight|ahead|strength|walking)\b/g) || []).length;
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

    // FIX: No aplicar la regla de número → suprimir si el prefijo numérico
    // es claramente una cantidad de pastillas (ej: "2 pills, 2 times a day")
    // La regla original quitaba el "2" al inicio cuando el resto empezaba con palabra EN,
    // lo que producía "pills, 2 times a day" — dejamos el número si va seguido de "pill/pills"
    const enStartWords = /^(or|before|after|the|was|were|is|are|have|had|do|does|did|when|where|what|how|why|which|that|this|it|in|of|for|with|a|an|and|but|not|no|any|all|one|two|three|four|some|your|their|our|my|its)/i;
    const shortPrefixMatch = t.match(/^(\d{1,3}\.?\s+)(\w.+)/);
    if (shortPrefixMatch && enStartWords.test(shortPrefixMatch[2])) {
      // Solo suprimir el número si lo que sigue NO es "pill(s)" o "tablet(s)"
      const nextWord = shortPrefixMatch[2].split(/\s+/)[0].toLowerCase();
      if (!/^pills?|tablets?$/.test(nextWord)) {
        t = shortPrefixMatch[2].charAt(0).toUpperCase() + shortPrefixMatch[2].slice(1);
      }
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

    const now = Date.now();
    if (now - buf.lastPartialEmitMs < 150) return;
    buf.lastPartialEmitMs = now;

    this.emit(session, { text: buf.text, language: buf.lang, isNewTurn: false, sessionId });
  }

  private async closeTurn(sessionId: string, reason: string): Promise<void> {
    const session = this.sessionData.get(sessionId);
    if (!session) return;
    const buf = session.buffer;

    // Usar buf.text como fuente primaria (el texto más reciente de AAI).
    // Solo usar peakText si buf.text está vacío o es claramente más corto por una
    // regresión de AAI (AAI volvió a un texto anterior). Criterio: si peakText
    // es >20% más largo que buf.text Y buf.text no termina en puntuación final,
    // es probable regresión — usar peak. Si buf.text termina en "?" o "." es
    // el texto correcto aunque sea más corto (AAI corrigió la frase).
    const bufHasTerminalPunct = /[.!?]$/.test((buf.text || '').trim());
    const peakIsSignificantlyLonger = buf.peakText &&
      buf.peakText.length > (buf.text?.length || 0) * 1.2 + 10;
    const textToClose = (peakIsSignificantlyLonger && !bufHasTerminalPunct)
      ? buf.peakText : (buf.text || buf.peakText);
    if (!textToClose) return;

    this.clearTimer(buf);
    const lang = buf.lang ?? this.detectLang(textToClose);
    const finalText = this.fixText(textToClose, lang);
    if (!finalText) { this.resetBuffer(buf); return; }

    const wordCount = finalText.trim().split(/\s+/).length;
    const isUniversalBackchannel = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(finalText.trim());
    const isNumericResponse = /^\d+\.?$/.test(finalText.trim());

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

    if (finalText.trim().length <= 2 && !isUniversalBackchannel && !isNumericResponse) {
      this.logger.log(`🚫 Ruido corto [${sessionId}]: "${finalText}"`);
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
    this.logger.log(`🎤 AssemblyAI u3-rt-pro iniciando [${sessionId}]`);

    const params = new URLSearchParams({
      sample_rate: '16000',
      format_turns: 'true',
      speech_model: 'u3-rt-pro',
    });

    const U3_PROMPT = 'Bilingual medical interpreter conversation. Doctor speaks English, patient speaks Spanish. Medical terminology includes seizures, Keppra, epilepsy, convulsiones, medicamentos. Do NOT translate — transcribe exactly as spoken in the original language.';

    const KEYTERMS = [
      'Keppra', 'convulsión', 'convulsiones', 'epilepsia',
      'seizure', 'seizures', 'levetiracetam', 'medicamento',
      'medicamentos', 'valproato', 'carbamazepina', 'lamotrigina',
      'cerebro', 'dosis', 'electroencefalograma', 'MRI',
    ];

    const WebSocket = require('ws');
    const ws = new WebSocket(
      `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`,
      { headers: { Authorization: apiKey } },
    );

    session.ws = ws;
    ws.on('open', () => this.logger.log(`✅ AssemblyAI u3-rt-pro abierto [${sessionId}]`));
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
        this.logger.log(`🔗 AAI u3-rt-pro [${sessionId}] sid=${msg.id}`);
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

        // ── Filtro de ruido: idioma no objetivo con baja confianza ────────────
        const isNonTargetLang = hasRealLang && aaiLang !== 'en' && aaiLang !== 'es';
        if (isNonTargetLang && aaiConf < 0.65 && wordCount <= 2 && !isUniversalWord) {
          this.logger.log(`🚫 Ruido [${aaiLang}=${aaiConf.toFixed(2)}] "${text}" [${sessionId}]`);
          return;
        }

        // ── Tipo A: artefactos de inicio de turno ────────────────────────────
        if (isFinal && !buf.text) {
          const tNorm = text.toLowerCase().replace(/[¿?!¡.,]/g, '').trim();

          // Artefactos conocidos explícitos
          const knownArtifacts = [
            'se despierta', 'qué hace como', 'es eso cómo', 'eso cómo',
            'hace como', 'es eso', 'as you may know', 'señor', 'i see him',
            'cómo hay', 'qué qué', 'qué es eso', 'qué', '¿qué',
          ];
          if (knownArtifacts.some(a => tNorm === a)) {
            this.logger.log(`🗑️ Artefacto conocido [${sessionId}]: "${text}"`);
            return;
          }

          // Pregunta genérica muy corta sin vocabulario médico/conversacional real
          if (wordCount >= 2 && wordCount <= 4) {
            const medicalOrCommon = /\b(convulsión|convulsiones|keppra|seizure|seizures|epilepsia|medicamento|dosis|dolor|espalda|cabeza|cerebro|hospital|doctor|médico|why|here|have|had|taking|your|you|when|last|how|many|what|were|before|after|increase|dose|missed|ever|day|days|sí|no|yes|okay|because|pero|desde|hace|tengo|tiene|tuve|dejé|pagar|cobrar|todos|días|años|meses|family|history|examine|examination|straight|follow|push|shoulder|extremities|strength|walking|pain|leg|back|run|down)\b/i;
            const isGenericArtifact = wordCount <= 3 && text.endsWith('?') && !medicalOrCommon.test(text);
            if (isGenericArtifact) {
              this.logger.log(`🗑️ Artefacto genérico [${sessionId}]: "${text}"`);
              return;
            }
          }
        }

        // ── Tipo B: subtítulos/overlay o repetición ──────────────────────────
        if (isFinal) {
          const sentences = text.split(/[.!?¿¡]+/).map(s => s.trim()).filter(Boolean);
          const normalized = sentences.map(s => s.toLowerCase().replace(/\s+/g, ' ').trim());
          const hasDuplicateSentence = new Set(normalized).size < normalized.length && sentences.length >= 2;
          const isMedicalList = sentences.length >= 3 && sentences.every(s => {
            const words = s.trim().split(/\s+/).filter(Boolean);
            return words.length <= 2 && /\b(seizure|seizures|epilepsy|epilepsia|convulsión|convulsiones|keppra|medication|medicamento)\b/i.test(s);
          });
          if (hasDuplicateSentence || isMedicalList) {
            this.logger.log(`🗑️ Subtítulo/overlay [${sessionId}]: "${text.substring(0, 60)}"`);
            return;
          }
        }

        // ── Tipo C: fragmentos incompletos o ruido corto ─────────────────────
        const endsWithDash = /[—–-]{1,2}$/.test(text.trim());
        const hasTerminalPunct = /[.!?]$/.test(text.trim());
        const isSingleMedicalWord = wordCount === 1 && /^(seizures?|epilepsy|epilepsia|convulsiones?|keppra|medication|medicamentos?)\.?$/i.test(text.trim());

        if (isFinal && (endsWithDash || isSingleMedicalWord) && !buf.text) {
          this.logger.log(`🗑️ Ruido suprimido [${sessionId}]: "${text}"`);
          return;
        }

        const isIncompleteFragment = isFinal && !hasTerminalPunct && !endsWithDash && wordCount <= 4 && !buf.text;
        if (isIncompleteFragment) {
          this.logger.log(`⏳ Fragmento incompleto silencioso [${sessionId}]: "${text}"`);
          buf.text = text;
          buf.peakText = text;

          // FIX: Detectar idioma correctamente ya en el fragmento inicial
          const { lang: fragLex, strong: fragStrong } = this.detectLangWithStrength(text);
          const startsObviouslyES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|y |no,|cuatro|tres|dos|uno)/i.test(text.trim());
          if (fragStrong || startsObviouslyES) {
            buf.lang = fragLex;
            buf.langConfident = fragStrong;
          } else {
            buf.lang = this.resolveLang(text, aaiLang, aaiConf, buf.lastEmittedLang, wordCount);
            buf.langConfident = hasRealLang;
          }

          buf.lastUpdateMs = now;
          buf.lastSeenText = text;
          buf.lastClosedMs = now;
          this.clearTimer(buf);
          buf.timer = setTimeout(() => {
            buf.timer = null;
            const sCheck = this.sessionData.get(sessionId);
            if (sCheck?.buffer.text === text) {
              this.logger.log(`🗑 Fragmento suprimido [${sessionId}]: "${text}"`);
              if (sCheck) this.resetBuffer(sCheck.buffer);
            }
          }, 1400);
          return;
        }

        // ── Corrección de idioma para texto español sin lang de AAI ──────────
        // FIX: El log era solo informativo pero no cambiaba el idioma efectivamente.
        // Ahora forzamos el idioma correcto ANTES de continuar.
        const startsObviouslyES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|y |no,|cuatro|tres|dos|uno)/i.test(text.trim());
        const { lang: preLex, strong: preStrong } = this.detectLangWithStrength(text);
        const forceES = !hasRealLang && (preStrong && preLex === 'es') || startsObviouslyES;

        if (isFinal && forceES && !buf.text) {
          this.logger.log(`🔧 LangCorrect forzado ES [${sessionId}]: "${text.substring(0, 50)}"`);
          buf.lang = 'es';
          buf.langConfident = preStrong;
        }

        // ── Tipo D: filtro de intérprete ─────────────────────────────────────
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

        // ── peakText solo crece con texto limpio ──────────────────────────────
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

        // Caso A: fragmento en buffer + llega texto completo → merge silencioso
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
            // Re-detectar idioma con el texto completo — el fragmento puede haber sido mal clasificado
            const { lang: mergeLex, strong: mergeStrong } = this.detectLangWithStrength(text);
            const startsES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|no,|cuatro|tres)/i.test(text.trim());
            if (hasRealLang) {
              buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
              buf.langConfident = true;
            } else if (mergeStrong || startsES) {
              buf.lang = startsES ? 'es' : mergeLex;
              buf.langConfident = mergeStrong;
            }
            // No return — continuar procesamiento para que el silence timer se actualice
          }
        }

        // Caso B: turno cerrado recientemente y el nuevo texto lo extiende/reemplaza
        // CRÍTICO: "Sí." cierra → llega "Sí, pero lo dejé." — debe reemplazar, no crear bloque nuevo
        if (!buf.text && msSinceClose < 800 && buf.lastEmittedText && msSinceForceClose >= 2000) {
          const lastNorm = normalize(buf.lastEmittedText);
          const newNorm2 = normalize(text);
          // Para textos cortos (≤3 palabras): usar todo el texto como prefijo de búsqueda
          // Para textos largos: usar los primeros 20 chars
          const lastWords = lastNorm.split(/\s+/).filter(Boolean).length;
          const prefixLen = lastWords <= 3 ? lastNorm.length : 20;
          const prefix = lastNorm.substring(0, prefixLen);
          if (prefix.length >= 2 && newNorm2.startsWith(prefix) && newNorm2.length > lastNorm.length) {
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

        // Caso C: fragmento muy corto (1 palabra) de idioma diferente → flush primero
        // Solo con 1 palabra porque con 2+ palabras puede ser una extensión legítima
        // del mismo hablante (ej: buf="Sí." → nuevo="Sí, doctor." es el mismo Turn)
        if (buf.text && buf.text.split(/\s+/).filter(Boolean).length === 1) {
          const bufNorm = normalize(buf.text);
          const newNorm = normalize(text);
          const isExtension = newNorm.startsWith(bufNorm.substring(0, Math.min(bufNorm.length, 10)));
          const isSameLang = buf.lang && this.resolveLang(text, aaiLang, aaiConf, buf.lang, wordCount) === buf.lang;
          if (!isExtension && !isSameLang) {
            this.logger.log(`🔀 FragmentFlush [${sessionId}]: "${buf.text}" → nuevo turno`);
            const oldText = buf.text;
            const oldLang = buf.lang;
            this.resetBuffer(buf);
            if (oldText) {
              buf.text = oldText;
              buf.lang = oldLang;
              await this.closeTurn(sessionId, 'fragmentFlush');
            }
          }
        }

        const { lang: lexLang, strong: lexStrong } = this.detectLangWithStrength(text);
        // Aumentar el gap requerido: 400ms causaba SpeakerChange dentro del mismo Turn del doctor.
        // Con 800ms solo se dispara cuando hay una pausa real entre hablantes.
        const silenceGap = now - buf.lastUpdateMs > 800;
        const bufEmpty = !buf.lang || !buf.text;

        // ── Asignación de idioma ──────────────────────────────────────────────
        if (isUniversalWord && buf.lastEmittedLang) {
          const isAmbiguousNo = /^no\.?,?$/i.test(text.trim());
          const isDefinitelySpanish = /^(sí|sí,|si,)$/i.test(text.trim());
          const isDefinitelyEnglish = /^(yes|yeah|nope)\.?,?$/i.test(text.trim());

          if (isDefinitelySpanish) {
            buf.lang = 'es';
            buf.langConfident = false;
          } else if (isDefinitelyEnglish) {
            buf.lang = 'en';
            buf.langConfident = false;
          } else if (isAmbiguousNo) {
            if (hasRealLang) {
              buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
              buf.langConfident = true;
              this.logger.log(`🔄 AmbiguousNo→AAI [${buf.lang}] conf=${aaiConf.toFixed(2)} [${sessionId}]`);
            } else {
              const recentLangs = s.conversationHistory.slice(-2).map(h => h.lang);
              const lastLang = recentLangs[recentLangs.length - 1] ?? buf.lastEmittedLang;
              buf.lang = lastLang === 'en' ? 'es' : 'en';
              buf.langConfident = false;
              this.logger.log(`🔄 AmbiguousNo→History [${buf.lastEmittedLang}→${buf.lang}] [${sessionId}]`);
            }
          } else {
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
        } else if (bufEmpty && !buf.lang) {
          // Solo asignar idioma si el buffer está completamente vacío (sin asignación previa del forceES)
          if (hasRealLang) {
            buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
            buf.langConfident = true;
            if (buf.lang !== buf.lastEmittedLang) {
              this.logger.log(`🌍 LangFromAAI [${buf.lastEmittedLang}→${buf.lang}] [${sessionId}]`);
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
        } else if (!bufEmpty && hasRealLang) {
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

        // ── Speaker change ────────────────────────────────────────────────────
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

        // ── Split de Turn mezclado EN + respuesta ES ──────────────────────────
        if (isFinal && buf.lang === 'en') {
          const mixedMatch = text.match(/^(.+\?)\s+((?:sí|si|no|claro|bien|okay|ok|cuatro|four|tres|three|dos|two|uno|one|\d+)[^?]*)$/i);
          if (mixedMatch) {
            const enPart = mixedMatch[1].trim();
            const esPart = mixedMatch[2].trim();
            const enWords = enPart.split(/\s+/).filter(Boolean);
            const esWords = esPart.split(/\s+/).filter(Boolean).length;
            const hasEnContent = enWords.some(w => /^(the|you|have|had|are|was|were|do|does|did|your|any|ever|since|before|after|how|when|what|why)$/i.test(w));
            if (enWords.length >= 4 && esWords <= 6 && hasEnContent) {
              this.logger.log(`✂️ Split EN+ES [${sessionId}]: EN="${enPart.substring(0, 50)}" ES="${esPart}"`);
              buf.text = enPart;
              buf.peakText = enPart;
              buf.lang = 'en';
              this.clearTimer(buf);
              await this.closeTurn(sessionId, 'splitMixed');
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

        // ── ForceClose por mezcla EN+ES ───────────────────────────────────────
        if (wordCount >= 8 && buf.text) {
          const words = text.trim().split(/\s+/);
          const esOnly = /^(que|los|las|del|una|con|para|pero|desde|hace|porque|también|cuando|como|esto|eso|fue|han|tengo|tuve|tenía|convulsiones|días|mes|año|años|siempre|nunca|alguna|dejé|pagar|cobraba|incrementaron|tomarla|todos|ninguno|manejar|pregunta|exámenes|resultados|familia)$/i;
          const enOnly = /^(the|and|you|have|had|are|taking|medications|seizures|since|before|after|dose|increase|missed|those|pills|times|every|medical|conditions|family|history|examine|when|was|your|last|seizure|not|examination|follow|straight|ahead|strength|walking|pain|leg|back)$/i;
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

        // ── Silence timer ─────────────────────────────────────────────────────
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
            // Turnos cortos cierran rápido, turnos largos esperan más
            const closeDelay = staleWords > 15 ? 800
              : staleWords > 8 ? 600
              : T_SILENCE_CLOSE_STALE;
            buf.timer = setTimeout(() => {
              buf.timer = null;
              const s2 = this.sessionData.get(sessionId);
              if (s2?.ws?.readyState === 1 && s2.buffer.text) {
                this.logger.log(`⚡ ForceEndpoint [${sessionId}] (${staleWords}w)`);
                s2.ws.send(JSON.stringify({ type: 'ForceEndpoint' }));
                const fb = setTimeout(() => {
                  this.logger.log(`⏱ Silence close fallback [${sessionId}]`);
                  this.closeTurn(sessionId, 'silence');
                }, 800);
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
      this.logger.log(`⏳ Cerrando AAI u3-rt-pro [${sessionId}]`);
      const s = this.sessionData.get(sessionId);
      if (s?.buffer.text || s?.buffer.peakText) await this.closeTurn(sessionId, 'userStop');
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'Terminate' }));
        await new Promise((r) => setTimeout(r, 2500));
      }
      ws.close();
      this.logger.log(`🛑 AAI u3-rt-pro cerrado [${sessionId}]`);
    };

    return { send, close };
  }

  private async claudePipeline(text: string, lang: 'es' | 'en', session: SessionData, sessionId: string) {
    const history = [...session.conversationHistory];
    const { result, correctedLang } = await this.correctWithClaude(text, lang, history);
    if (result !== text || correctedLang !== lang) {
      this.logger.log(`✨ CLAUDE [${lang}→${correctedLang}]: "${result.substring(0, 80)}"`);
      const idx = session.conversationHistory.findLastIndex((t) => t.text === text);
      if (idx >= 0) {
        session.conversationHistory[idx].text = result;
        session.conversationHistory[idx].lang = correctedLang;
      }
      if (correctedLang !== lang && session.buffer.lastEmittedLang === lang) {
        session.buffer.lastEmittedLang = correctedLang;
      }
      // Emitir siempre con el originalText para que el frontend pueda reemplazar
      // el bloque correcto sin dejar el bloque viejo visible
      this.emit(session, {
        text: result,
        language: correctedLang,
        isCorrection: true,
        originalText: text,
        sessionId,
      });
    }
  }

  private async correctWithClaude(
    text: string, lang: 'es' | 'en', _history: ConversationTurn[],
  ): Promise<{ result: string; correctedLang: 'es' | 'en' }> {
    // Claude desactivado — causaba eliminación de texto válido y correcciones incorrectas.
    // fixText() ya maneja Keppra y 2,000. Solo aplicar correcciones 100% deterministas aquí.

    let result = text;

    // 1. Keppra (fixText ya lo hace, pero por si llega aquí antes)
    result = result.replace(/\b(keprah?|kepra|quepra|kephra|kebri[ah]?|kebra)\b/gi, 'Keppra');

    // 2. "Sí, 2000." → "Sí, 2,000." SOLO si el 2000 está solo como número de dosis
    //    NO cambiar "2 pills", "2 times", "2 days" — solo el número puro 2000
    result = result.replace(/\b2000\b/g, '2,000');

    // 3. "si " al inicio → "Sí, " (solo si no tiene tilde ya)
    result = result.replace(/^si\s/i, (m) => m[0] === 'S' ? 'Sí, ' : 'Sí, ');

    if (this.norm(result) === this.norm(text)) return { result: text, correctedLang: lang };
    const correctedLang: 'es' | 'en' = this.detectLang(result) ?? lang;
    return { result, correctedLang };
  }
}

// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';
// import Anthropic from '@anthropic-ai/sdk';

// const T_SILENCE_CLOSE = 1500;        // 1.5s: balance entre fluidez y no cortar mid-sentence
// const T_SILENCE_CLOSE_STALE = 800;  // 0.8s: cuando el texto está estancado, cerrar más rápido
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
//     const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar|manejar|dolor|espalda|pregunta|exámenes|resultados|familia|ninguno)\b/g) || []).length;
//     const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes|examination|follow|straight|ahead|strength|walking)\b/g) || []).length;
//     return esScore > enScore ? 'es' : 'en';
//   }

//   private detectLangWithStrength(text: string): { lang: 'es' | 'en'; strong: boolean } {
//     const t = text.toLowerCase();
//     const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar|manejar|dolor|espalda|pregunta|exámenes|resultados|familia|ninguno)\b/g) || []).length;
//     const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes|examination|follow|straight|ahead|strength|walking)\b/g) || []).length;
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

//     // FIX: No aplicar la regla de número → suprimir si el prefijo numérico
//     // es claramente una cantidad de pastillas (ej: "2 pills, 2 times a day")
//     // La regla original quitaba el "2" al inicio cuando el resto empezaba con palabra EN,
//     // lo que producía "pills, 2 times a day" — dejamos el número si va seguido de "pill/pills"
//     const enStartWords = /^(or|before|after|the|was|were|is|are|have|had|do|does|did|when|where|what|how|why|which|that|this|it|in|of|for|with|a|an|and|but|not|no|any|all|one|two|three|four|some|your|their|our|my|its)/i;
//     const shortPrefixMatch = t.match(/^(\d{1,3}\.?\s+)(\w.+)/);
//     if (shortPrefixMatch && enStartWords.test(shortPrefixMatch[2])) {
//       // Solo suprimir el número si lo que sigue NO es "pill(s)" o "tablet(s)"
//       const nextWord = shortPrefixMatch[2].split(/\s+/)[0].toLowerCase();
//       if (!/^pills?|tablets?$/.test(nextWord)) {
//         t = shortPrefixMatch[2].charAt(0).toUpperCase() + shortPrefixMatch[2].slice(1);
//       }
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

//     const now = Date.now();
//     if (now - buf.lastPartialEmitMs < 150) return;
//     buf.lastPartialEmitMs = now;

//     this.emit(session, { text: buf.text, language: buf.lang, isNewTurn: false, sessionId });
//   }

//   private async closeTurn(sessionId: string, reason: string): Promise<void> {
//     const session = this.sessionData.get(sessionId);
//     if (!session) return;
//     const buf = session.buffer;

//     // Usar buf.text como fuente primaria (el texto más reciente de AAI).
//     // Solo usar peakText si buf.text está vacío o es claramente más corto por una
//     // regresión de AAI (AAI volvió a un texto anterior). Criterio: si peakText
//     // es >20% más largo que buf.text Y buf.text no termina en puntuación final,
//     // es probable regresión — usar peak. Si buf.text termina en "?" o "." es
//     // el texto correcto aunque sea más corto (AAI corrigió la frase).
//     const bufHasTerminalPunct = /[.!?]$/.test((buf.text || '').trim());
//     const peakIsSignificantlyLonger = buf.peakText &&
//       buf.peakText.length > (buf.text?.length || 0) * 1.2 + 10;
//     const textToClose = (peakIsSignificantlyLonger && !bufHasTerminalPunct)
//       ? buf.peakText : (buf.text || buf.peakText);
//     if (!textToClose) return;

//     this.clearTimer(buf);
//     const lang = buf.lang ?? this.detectLang(textToClose);
//     const finalText = this.fixText(textToClose, lang);
//     if (!finalText) { this.resetBuffer(buf); return; }

//     const wordCount = finalText.trim().split(/\s+/).length;
//     const isUniversalBackchannel = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(finalText.trim());
//     const isNumericResponse = /^\d+\.?$/.test(finalText.trim());

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

//     if (finalText.trim().length <= 2 && !isUniversalBackchannel && !isNumericResponse) {
//       this.logger.log(`🚫 Ruido corto [${sessionId}]: "${finalText}"`);
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
//     this.logger.log(`🎤 AssemblyAI u3-rt-pro iniciando [${sessionId}]`);

//     const params = new URLSearchParams({
//       sample_rate: '16000',
//       format_turns: 'true',
//       speech_model: 'u3-rt-pro',
//     });

//     const U3_PROMPT = 'Bilingual medical interpreter conversation. Doctor speaks English, patient speaks Spanish. Medical terminology includes seizures, Keppra, epilepsy, convulsiones, medicamentos. Do NOT translate — transcribe exactly as spoken in the original language.';

//     const KEYTERMS = [
//       'Keppra', 'convulsión', 'convulsiones', 'epilepsia',
//       'seizure', 'seizures', 'levetiracetam', 'medicamento',
//       'medicamentos', 'valproato', 'carbamazepina', 'lamotrigina',
//       'cerebro', 'dosis', 'electroencefalograma', 'MRI',
//     ];

//     const WebSocket = require('ws');
//     const ws = new WebSocket(
//       `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`,
//       { headers: { Authorization: apiKey } },
//     );

//     session.ws = ws;
//     ws.on('open', () => this.logger.log(`✅ AssemblyAI u3-rt-pro abierto [${sessionId}]`));
//     ws.on('error', (err: Error) => this.logger.error(`❌ AAI error [${sessionId}]: ${err.message}`));

//     ws.on('close', (code: number) => {
//       this.logger.log(`🔒 AAI cerrado [${sessionId}] (${code})`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) this.closeTurn(sessionId, 'streamClose');
//       this.sessionData.delete(sessionId);
//     });

//     ws.on('message', async (raw: any) => {
//       const s = this.sessionData.get(sessionId);
//       if (!s) return;
//       let msg: any;
//       try { msg = JSON.parse(raw.toString()); } catch { return; }

//       const buf = s.buffer;
//       const now = Date.now();

//       if (msg.type === 'Begin') {
//         this.logger.log(`🔗 AAI u3-rt-pro [${sessionId}] sid=${msg.id}`);
//         ws.send(JSON.stringify({ type: 'UpdateConfiguration', keyterms: KEYTERMS, prompt: U3_PROMPT }));
//         return;
//       }

//       if (msg.type === 'Turn') {
//         const text: string = (msg.transcript || '').trim();
//         const aaiLang: string = msg.language_code ?? '';
//         const aaiConf: number = msg.language_confidence ?? 0;
//         const isFinal: boolean = msg.turn_is_formatted === true;
//         const wordCount = text.split(/\s+/).filter(Boolean).length;

//         const hasRealLang = !!(aaiLang && aaiLang !== 'undefined' && aaiConf > 0);
//         const isUniversalWord = /^(no|sí|si|yes|ok|yeah|bien|\d+)\.?,?$/i.test(text.trim());

//         if (isFinal && (s as any)._forceEndpointFallback) {
//           clearTimeout((s as any)._forceEndpointFallback);
//           delete (s as any)._forceEndpointFallback;
//         }

//         this.logger.log(`🔬 RAW fmt=${isFinal} lang=${aaiLang} conf=${aaiConf.toFixed(2)} "${text.substring(0, 60)}" [${sessionId}]`);
//         if (!text) return;

//         // ── Filtro de ruido: idioma no objetivo con baja confianza ────────────
//         const isNonTargetLang = hasRealLang && aaiLang !== 'en' && aaiLang !== 'es';
//         if (isNonTargetLang && aaiConf < 0.65 && wordCount <= 2 && !isUniversalWord) {
//           this.logger.log(`🚫 Ruido [${aaiLang}=${aaiConf.toFixed(2)}] "${text}" [${sessionId}]`);
//           return;
//         }

//         // ── Tipo A: artefactos de inicio de turno ────────────────────────────
//         if (isFinal && !buf.text) {
//           const tNorm = text.toLowerCase().replace(/[¿?!¡.,]/g, '').trim();

//           // Artefactos conocidos explícitos
//           const knownArtifacts = [
//             'se despierta', 'qué hace como', 'es eso cómo', 'eso cómo',
//             'hace como', 'es eso', 'as you may know', 'señor', 'i see him',
//             'cómo hay', 'qué qué', 'qué es eso', 'qué', '¿qué',
//           ];
//           if (knownArtifacts.some(a => tNorm === a)) {
//             this.logger.log(`🗑️ Artefacto conocido [${sessionId}]: "${text}"`);
//             return;
//           }

//           // Pregunta genérica muy corta sin vocabulario médico/conversacional real
//           if (wordCount >= 2 && wordCount <= 4) {
//             const medicalOrCommon = /\b(convulsión|convulsiones|keppra|seizure|seizures|epilepsia|medicamento|dosis|dolor|espalda|cabeza|cerebro|hospital|doctor|médico|why|here|have|had|taking|your|you|when|last|how|many|what|were|before|after|increase|dose|missed|ever|day|days|sí|no|yes|okay|because|pero|desde|hace|tengo|tiene|tuve|dejé|pagar|cobrar|todos|días|años|meses|family|history|examine|examination|straight|follow|push|shoulder|extremities|strength|walking|pain|leg|back|run|down)\b/i;
//             const isGenericArtifact = wordCount <= 3 && text.endsWith('?') && !medicalOrCommon.test(text);
//             if (isGenericArtifact) {
//               this.logger.log(`🗑️ Artefacto genérico [${sessionId}]: "${text}"`);
//               return;
//             }
//           }
//         }

//         // ── Tipo B: subtítulos/overlay o repetición ──────────────────────────
//         if (isFinal) {
//           const sentences = text.split(/[.!?¿¡]+/).map(s => s.trim()).filter(Boolean);
//           const normalized = sentences.map(s => s.toLowerCase().replace(/\s+/g, ' ').trim());
//           const hasDuplicateSentence = new Set(normalized).size < normalized.length && sentences.length >= 2;
//           const isMedicalList = sentences.length >= 3 && sentences.every(s => {
//             const words = s.trim().split(/\s+/).filter(Boolean);
//             return words.length <= 2 && /\b(seizure|seizures|epilepsy|epilepsia|convulsión|convulsiones|keppra|medication|medicamento)\b/i.test(s);
//           });
//           if (hasDuplicateSentence || isMedicalList) {
//             this.logger.log(`🗑️ Subtítulo/overlay [${sessionId}]: "${text.substring(0, 60)}"`);
//             return;
//           }
//         }

//         // ── Tipo C: fragmentos incompletos o ruido corto ─────────────────────
//         const endsWithDash = /[—–-]{1,2}$/.test(text.trim());
//         const hasTerminalPunct = /[.!?]$/.test(text.trim());
//         const isSingleMedicalWord = wordCount === 1 && /^(seizures?|epilepsy|epilepsia|convulsiones?|keppra|medication|medicamentos?)\.?$/i.test(text.trim());

//         if (isFinal && (endsWithDash || isSingleMedicalWord) && !buf.text) {
//           this.logger.log(`🗑️ Ruido suprimido [${sessionId}]: "${text}"`);
//           return;
//         }

//         const isIncompleteFragment = isFinal && !hasTerminalPunct && !endsWithDash && wordCount <= 4 && !buf.text;
//         if (isIncompleteFragment) {
//           this.logger.log(`⏳ Fragmento incompleto silencioso [${sessionId}]: "${text}"`);
//           buf.text = text;
//           buf.peakText = text;

//           // FIX: Detectar idioma correctamente ya en el fragmento inicial
//           const { lang: fragLex, strong: fragStrong } = this.detectLangWithStrength(text);
//           const startsObviouslyES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|y |no,|cuatro|tres|dos|uno)/i.test(text.trim());
//           if (fragStrong || startsObviouslyES) {
//             buf.lang = fragLex;
//             buf.langConfident = fragStrong;
//           } else {
//             buf.lang = this.resolveLang(text, aaiLang, aaiConf, buf.lastEmittedLang, wordCount);
//             buf.langConfident = hasRealLang;
//           }

//           buf.lastUpdateMs = now;
//           buf.lastSeenText = text;
//           buf.lastClosedMs = now;
//           this.clearTimer(buf);
//           buf.timer = setTimeout(() => {
//             buf.timer = null;
//             const sCheck = this.sessionData.get(sessionId);
//             if (sCheck?.buffer.text === text) {
//               this.logger.log(`🗑 Fragmento suprimido [${sessionId}]: "${text}"`);
//               if (sCheck) this.resetBuffer(sCheck.buffer);
//             }
//           }, 1400);
//           return;
//         }

//         // ── Corrección de idioma para texto español sin lang de AAI ──────────
//         // FIX: El log era solo informativo pero no cambiaba el idioma efectivamente.
//         // Ahora forzamos el idioma correcto ANTES de continuar.
//         const startsObviouslyES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|y |no,|cuatro|tres|dos|uno)/i.test(text.trim());
//         const { lang: preLex, strong: preStrong } = this.detectLangWithStrength(text);
//         const forceES = !hasRealLang && (preStrong && preLex === 'es') || startsObviouslyES;

//         if (isFinal && forceES && !buf.text) {
//           this.logger.log(`🔧 LangCorrect forzado ES [${sessionId}]: "${text.substring(0, 50)}"`);
//           buf.lang = 'es';
//           buf.langConfident = preStrong;
//         }

//         // ── Tipo D: filtro de intérprete ─────────────────────────────────────
//         if (isFinal) {
//           const tLow = text.toLowerCase().replace(/[¿?!¡.,]/g, '').trim();
//           const interpreterPatterns = [
//             /^cómo hay (muchos?|muchas?) convulsiones/,
//             /^así un medicamento/,
//             /^es antes o después de la dosis/,
//             /^tú tienes pastillas/,
//             /^qué fueron ustedes/,
//             /^estás tomando keppra/,
//             /^y hace cuánto tiempo tiene convulsiones/,
//             /^cuándo fue su últ[io]m[ao] convulsión/,
//             /^es eso lo que toma ahora/,
//             /^si alguna vez (ha|has|he) dejado de tomarla/,
//             /^hace cuánto tiempo tiene/,
//           ];
//           const isKnownInterpreter = interpreterPatterns.some(p => p.test(tLow));

//           const msSinceLastClose = now - buf.lastClosedMs;
//           const lastWasEn = buf.lastEmittedLang === 'en';
//           const isQuickEsAfterEn = lastWasEn && msSinceLastClose < 2000 && !buf.text;
//           const detectLangHere = this.resolveLang(text, aaiLang, aaiConf, buf.lastEmittedLang, wordCount);
//           let isSemanticInterpreter = false;
//           if (isQuickEsAfterEn && detectLangHere === 'es' && wordCount <= 10 && /[?]$/.test(text.trim())) {
//             const medTerms = /\b(convulsión|convulsiones|seizure|seizures|keppra|medicamento|medicamentos|dosis|dose|abril|april|junio|june|pastilla|pill|tomando|taking|dejó|stopped|cuánto|cuándo|when|antes|before|después|after|aumento|increase)\b/i;
//             const lastEnText = (buf.lastEmittedText || '').toLowerCase();
//             isSemanticInterpreter = medTerms.test(text) && medTerms.test(lastEnText);
//           }

//           if (isKnownInterpreter || isSemanticInterpreter) {
//             this.logger.log(`🎭 Intérprete filtrado [${sessionId}] (${isKnownInterpreter ? 'pattern' : 'semantic'} +${now - buf.lastClosedMs}ms): "${text}"`);
//             return;
//           }
//         }

//         // ── peakText solo crece con texto limpio ──────────────────────────────
//         const prevPeak = buf.peakText || '';
//         const isCleanGrowth = text.startsWith(prevPeak.substring(0, Math.min(prevPeak.length, 15)));
//         if (text.length > prevPeak.length && (isCleanGrowth || prevPeak.length < 10)) {
//           buf.peakText = text;
//         }

//         // ── Guard de continuación post-close ──────────────────────────────────
//         const msSinceClose = now - buf.lastClosedMs;
//         const msSinceForceClose = now - buf.forceClosedMs;

//         const normalize = (str: string) =>
//           str.replace(/Keppra/gi, 'kepra').replace(/[,\.!?¿¡—–]/g, '').replace(/\s+/g, ' ').trim().toLowerCase();

//         // Caso A: fragmento en buffer + llega texto completo → merge silencioso
//         if (buf.text && buf.text.split(/\s+/).length <= 4) {
//           const bufNorm = normalize(buf.text);
//           const newNorm = normalize(text);
//           if (newNorm.startsWith(bufNorm.substring(0, Math.min(bufNorm.length, 12))) && newNorm.length > bufNorm.length) {
//             this.logger.log(`🔁 FragmentMerge [${sessionId}]: "${buf.text}" → "${text.substring(0, 60)}"`);
//             this.clearTimer(buf);
//             buf.text = text;
//             buf.peakText = text;
//             buf.lastUpdateMs = now;
//             buf.lastSeenText = text;
//           }
//         }

//         // Caso B: turno cerrado recientemente y el nuevo texto lo extiende
//         if (!buf.text && msSinceClose < 1500 && buf.lastEmittedText && msSinceForceClose >= 2000) {
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

//         // Caso C: fragmento muy corto (1 palabra) de idioma diferente → flush primero
//         // Solo con 1 palabra porque con 2+ palabras puede ser una extensión legítima
//         // del mismo hablante (ej: buf="Sí." → nuevo="Sí, doctor." es el mismo Turn)
//         if (buf.text && buf.text.split(/\s+/).filter(Boolean).length === 1) {
//           const bufNorm = normalize(buf.text);
//           const newNorm = normalize(text);
//           const isExtension = newNorm.startsWith(bufNorm.substring(0, Math.min(bufNorm.length, 10)));
//           const isSameLang = buf.lang && this.resolveLang(text, aaiLang, aaiConf, buf.lang, wordCount) === buf.lang;
//           if (!isExtension && !isSameLang) {
//             this.logger.log(`🔀 FragmentFlush [${sessionId}]: "${buf.text}" → nuevo turno`);
//             const oldText = buf.text;
//             const oldLang = buf.lang;
//             this.resetBuffer(buf);
//             if (oldText) {
//               buf.text = oldText;
//               buf.lang = oldLang;
//               await this.closeTurn(sessionId, 'fragmentFlush');
//             }
//           }
//         }

//         const { lang: lexLang, strong: lexStrong } = this.detectLangWithStrength(text);
//         // Aumentar el gap requerido: 400ms causaba SpeakerChange dentro del mismo Turn del doctor.
//         // Con 800ms solo se dispara cuando hay una pausa real entre hablantes.
//         const silenceGap = now - buf.lastUpdateMs > 800;
//         const bufEmpty = !buf.lang || !buf.text;

//         // ── Asignación de idioma ──────────────────────────────────────────────
//         if (isUniversalWord && buf.lastEmittedLang) {
//           const isAmbiguousNo = /^no\.?,?$/i.test(text.trim());
//           const isDefinitelySpanish = /^(sí|sí,|si,)$/i.test(text.trim());
//           const isDefinitelyEnglish = /^(yes|yeah|nope)\.?,?$/i.test(text.trim());

//           if (isDefinitelySpanish) {
//             buf.lang = 'es';
//             buf.langConfident = false;
//           } else if (isDefinitelyEnglish) {
//             buf.lang = 'en';
//             buf.langConfident = false;
//           } else if (isAmbiguousNo) {
//             if (hasRealLang) {
//               buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
//               buf.langConfident = true;
//               this.logger.log(`🔄 AmbiguousNo→AAI [${buf.lang}] conf=${aaiConf.toFixed(2)} [${sessionId}]`);
//             } else {
//               const recentLangs = s.conversationHistory.slice(-2).map(h => h.lang);
//               const lastLang = recentLangs[recentLangs.length - 1] ?? buf.lastEmittedLang;
//               buf.lang = lastLang === 'en' ? 'es' : 'en';
//               buf.langConfident = false;
//               this.logger.log(`🔄 AmbiguousNo→History [${buf.lastEmittedLang}→${buf.lang}] [${sessionId}]`);
//             }
//           } else {
//             const isSpanishResponse = /^(sí|si)\.?,?$/i.test(text.trim());
//             if (isSpanishResponse && buf.lastEmittedLang === 'es') {
//               buf.lang = 'es';
//               buf.langConfident = false;
//             } else {
//               const opposite = buf.lastEmittedLang === 'en' ? 'es' : 'en';
//               buf.lang = opposite;
//               buf.langConfident = false;
//               this.logger.log(`🔄 UniversalFlip [${buf.lastEmittedLang}→${opposite}] "${text}" [${sessionId}]`);
//             }
//           }
//         } else if (bufEmpty && !buf.lang) {
//           // Solo asignar idioma si el buffer está completamente vacío (sin asignación previa del forceES)
//           if (hasRealLang) {
//             buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
//             buf.langConfident = true;
//             if (buf.lang !== buf.lastEmittedLang) {
//               this.logger.log(`🌍 LangFromAAI [${buf.lastEmittedLang}→${buf.lang}] [${sessionId}]`);
//             }
//           } else if (lexStrong && buf.lastEmittedLang && buf.lastEmittedLang !== lexLang) {
//             buf.lang = lexLang;
//             buf.langConfident = true;
//             this.logger.log(`🌍 LangFromLex [${buf.lastEmittedLang}→${lexLang}] [${sessionId}]`);
//           } else if (!lexStrong && this.isBackchannel(text) && buf.lastEmittedLang) {
//             const isSpanishBackchannel = /^(sí|si)\.?,?$/i.test(text.trim());
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
//         } else if (!bufEmpty && hasRealLang) {
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

//         // ── Speaker change ────────────────────────────────────────────────────
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

//         // ── Split de Turn mezclado EN + respuesta ES ──────────────────────────
//         if (isFinal && buf.lang === 'en') {
//           const mixedMatch = text.match(/^(.+\?)\s+((?:sí|si|no|claro|bien|okay|ok|cuatro|four|tres|three|dos|two|uno|one|\d+)[^?]*)$/i);
//           if (mixedMatch) {
//             const enPart = mixedMatch[1].trim();
//             const esPart = mixedMatch[2].trim();
//             const enWords = enPart.split(/\s+/).filter(Boolean);
//             const esWords = esPart.split(/\s+/).filter(Boolean).length;
//             const hasEnContent = enWords.some(w => /^(the|you|have|had|are|was|were|do|does|did|your|any|ever|since|before|after|how|when|what|why)$/i.test(w));
//             if (enWords.length >= 4 && esWords <= 6 && hasEnContent) {
//               this.logger.log(`✂️ Split EN+ES [${sessionId}]: EN="${enPart.substring(0, 50)}" ES="${esPart}"`);
//               buf.text = enPart;
//               buf.peakText = enPart;
//               buf.lang = 'en';
//               this.clearTimer(buf);
//               await this.closeTurn(sessionId, 'splitMixed');
//               const sAfterSplit = this.sessionData.get(sessionId);
//               if (sAfterSplit) {
//                 sAfterSplit.buffer.text = esPart;
//                 sAfterSplit.buffer.peakText = esPart;
//                 sAfterSplit.buffer.lang = 'es';
//                 sAfterSplit.buffer.langConfident = false;
//                 sAfterSplit.buffer.lastUpdateMs = now;
//                 await this.closeTurn(sessionId, 'splitMixed');
//               }
//               return;
//             }
//           }
//         }

//         // ── ForceClose por mezcla EN+ES ───────────────────────────────────────
//         if (wordCount >= 8 && buf.text) {
//           const words = text.trim().split(/\s+/);
//           const esOnly = /^(que|los|las|del|una|con|para|pero|desde|hace|porque|también|cuando|como|esto|eso|fue|han|tengo|tuve|tenía|convulsiones|días|mes|año|años|siempre|nunca|alguna|dejé|pagar|cobraba|incrementaron|tomarla|todos|ninguno|manejar|pregunta|exámenes|resultados|familia)$/i;
//           const enOnly = /^(the|and|you|have|had|are|taking|medications|seizures|since|before|after|dose|increase|missed|those|pills|times|every|medical|conditions|family|history|examine|when|was|your|last|seizure|not|examination|follow|straight|ahead|strength|walking|pain|leg|back)$/i;
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

//         // ── Silence timer ─────────────────────────────────────────────────────
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
//             // Dar más tiempo a turnos largos: el doctor a veces hace pausas mid-sentence
//             // en frases de 10+ palabras. Con poco tiempo se cierra antes de terminar.
//             const closeDelay = staleWords > 15 ? 1200
//               : staleWords > 8 ? 900
//               : T_SILENCE_CLOSE_STALE;
//             buf.timer = setTimeout(() => {
//               buf.timer = null;
//               const s2 = this.sessionData.get(sessionId);
//               if (s2?.ws?.readyState === 1 && s2.buffer.text) {
//                 this.logger.log(`⚡ ForceEndpoint [${sessionId}] (${staleWords}w)`);
//                 s2.ws.send(JSON.stringify({ type: 'ForceEndpoint' }));
//                 const fb = setTimeout(() => {
//                   this.logger.log(`⏱ Silence close fallback [${sessionId}]`);
//                   this.closeTurn(sessionId, 'silence');
//                 }, 800);
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
//       this.logger.log(`⏳ Cerrando AAI u3-rt-pro [${sessionId}]`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) await this.closeTurn(sessionId, 'userStop');
//       if (ws.readyState === WebSocket.OPEN) {
//         ws.send(JSON.stringify({ type: 'Terminate' }));
//         await new Promise((r) => setTimeout(r, 2500));
//       }
//       ws.close();
//       this.logger.log(`🛑 AAI u3-rt-pro cerrado [${sessionId}]`);
//     };

//     return { send, close };
//   }

//   private async claudePipeline(text: string, lang: 'es' | 'en', session: SessionData, sessionId: string) {
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
//       // Emitir siempre con el originalText para que el frontend pueda reemplazar
//       // el bloque correcto sin dejar el bloque viejo visible
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
//     text: string, lang: 'es' | 'en', _history: ConversationTurn[],
//   ): Promise<{ result: string; correctedLang: 'es' | 'en' }> {
//     // Claude desactivado — causaba eliminación de texto válido y correcciones incorrectas.
//     // fixText() ya maneja Keppra y 2,000. Solo aplicar correcciones 100% deterministas aquí.

//     let result = text;

//     // 1. Keppra (fixText ya lo hace, pero por si llega aquí antes)
//     result = result.replace(/\b(keprah?|kepra|quepra|kephra|kebri[ah]?|kebra)\b/gi, 'Keppra');

//     // 2. "Sí, 2000." → "Sí, 2,000." SOLO si el 2000 está solo como número de dosis
//     //    NO cambiar "2 pills", "2 times", "2 days" — solo el número puro 2000
//     result = result.replace(/\b2000\b/g, '2,000');

//     // 3. "si " al inicio → "Sí, " (solo si no tiene tilde ya)
//     result = result.replace(/^si\s/i, (m) => m[0] === 'S' ? 'Sí, ' : 'Sí, ');

//     if (this.norm(result) === this.norm(text)) return { result: text, correctedLang: lang };
//     const correctedLang: 'es' | 'en' = this.detectLang(result) ?? lang;
//     return { result, correctedLang };
//   }
// }
// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';
// import Anthropic from '@anthropic-ai/sdk';

// const T_SILENCE_CLOSE = 1500;        // 1.5s: balance entre fluidez y no cortar mid-sentence
// const T_SILENCE_CLOSE_STALE = 800;  // 0.8s: cuando el texto está estancado, cerrar más rápido
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
//     const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar|manejar|dolor|espalda|pregunta|exámenes|resultados|familia|ninguno)\b/g) || []).length;
//     const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes|examination|follow|straight|ahead|strength|walking)\b/g) || []).length;
//     return esScore > enScore ? 'es' : 'en';
//   }

//   private detectLangWithStrength(text: string): { lang: 'es' | 'en'; strong: boolean } {
//     const t = text.toLowerCase();
//     const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar|manejar|dolor|espalda|pregunta|exámenes|resultados|familia|ninguno)\b/g) || []).length;
//     const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes|examination|follow|straight|ahead|strength|walking)\b/g) || []).length;
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

//     // FIX: No aplicar la regla de número → suprimir si el prefijo numérico
//     // es claramente una cantidad de pastillas (ej: "2 pills, 2 times a day")
//     // La regla original quitaba el "2" al inicio cuando el resto empezaba con palabra EN,
//     // lo que producía "pills, 2 times a day" — dejamos el número si va seguido de "pill/pills"
//     const enStartWords = /^(or|before|after|the|was|were|is|are|have|had|do|does|did|when|where|what|how|why|which|that|this|it|in|of|for|with|a|an|and|but|not|no|any|all|one|two|three|four|some|your|their|our|my|its)/i;
//     const shortPrefixMatch = t.match(/^(\d{1,3}\.?\s+)(\w.+)/);
//     if (shortPrefixMatch && enStartWords.test(shortPrefixMatch[2])) {
//       // Solo suprimir el número si lo que sigue NO es "pill(s)" o "tablet(s)"
//       const nextWord = shortPrefixMatch[2].split(/\s+/)[0].toLowerCase();
//       if (!/^pills?|tablets?$/.test(nextWord)) {
//         t = shortPrefixMatch[2].charAt(0).toUpperCase() + shortPrefixMatch[2].slice(1);
//       }
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

//     const now = Date.now();
//     if (now - buf.lastPartialEmitMs < 150) return;
//     buf.lastPartialEmitMs = now;

//     this.emit(session, { text: buf.text, language: buf.lang, isNewTurn: false, sessionId });
//   }

//   private async closeTurn(sessionId: string, reason: string): Promise<void> {
//     const session = this.sessionData.get(sessionId);
//     if (!session) return;
//     const buf = session.buffer;

//     // Usar buf.text como fuente primaria (el texto más reciente de AAI).
//     // Solo usar peakText si buf.text está vacío o es claramente más corto por una
//     // regresión de AAI (AAI volvió a un texto anterior). Criterio: si peakText
//     // es >20% más largo que buf.text Y buf.text no termina en puntuación final,
//     // es probable regresión — usar peak. Si buf.text termina en "?" o "." es
//     // el texto correcto aunque sea más corto (AAI corrigió la frase).
//     const bufHasTerminalPunct = /[.!?]$/.test((buf.text || '').trim());
//     const peakIsSignificantlyLonger = buf.peakText &&
//       buf.peakText.length > (buf.text?.length || 0) * 1.2 + 10;
//     const textToClose = (peakIsSignificantlyLonger && !bufHasTerminalPunct)
//       ? buf.peakText : (buf.text || buf.peakText);
//     if (!textToClose) return;

//     this.clearTimer(buf);
//     const lang = buf.lang ?? this.detectLang(textToClose);
//     const finalText = this.fixText(textToClose, lang);
//     if (!finalText) { this.resetBuffer(buf); return; }

//     const wordCount = finalText.trim().split(/\s+/).length;
//     const isUniversalBackchannel = /^(no|sí|si|yes|ok|yeah|bien)\.?,?$/i.test(finalText.trim());
//     const isNumericResponse = /^\d+\.?$/.test(finalText.trim());

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

//     if (finalText.trim().length <= 2 && !isUniversalBackchannel && !isNumericResponse) {
//       this.logger.log(`🚫 Ruido corto [${sessionId}]: "${finalText}"`);
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
//     this.logger.log(`🎤 AssemblyAI u3-rt-pro iniciando [${sessionId}]`);

//     const params = new URLSearchParams({
//       sample_rate: '16000',
//       format_turns: 'true',
//       speech_model: 'u3-rt-pro',
//     });

//     const U3_PROMPT = 'Bilingual medical interpreter conversation. Doctor speaks English, patient speaks Spanish. Medical terminology includes seizures, Keppra, epilepsy, convulsiones, medicamentos. Do NOT translate — transcribe exactly as spoken in the original language.';

//     const KEYTERMS = [
//       'Keppra', 'convulsión', 'convulsiones', 'epilepsia',
//       'seizure', 'seizures', 'levetiracetam', 'medicamento',
//       'medicamentos', 'valproato', 'carbamazepina', 'lamotrigina',
//       'cerebro', 'dosis', 'electroencefalograma', 'MRI',
//     ];

//     const WebSocket = require('ws');
//     const ws = new WebSocket(
//       `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`,
//       { headers: { Authorization: apiKey } },
//     );

//     session.ws = ws;
//     ws.on('open', () => this.logger.log(`✅ AssemblyAI u3-rt-pro abierto [${sessionId}]`));
//     ws.on('error', (err: Error) => this.logger.error(`❌ AAI error [${sessionId}]: ${err.message}`));

//     ws.on('close', (code: number) => {
//       this.logger.log(`🔒 AAI cerrado [${sessionId}] (${code})`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) this.closeTurn(sessionId, 'streamClose');
//       this.sessionData.delete(sessionId);
//     });

//     ws.on('message', async (raw: any) => {
//       const s = this.sessionData.get(sessionId);
//       if (!s) return;
//       let msg: any;
//       try { msg = JSON.parse(raw.toString()); } catch { return; }

//       const buf = s.buffer;
//       const now = Date.now();

//       if (msg.type === 'Begin') {
//         this.logger.log(`🔗 AAI u3-rt-pro [${sessionId}] sid=${msg.id}`);
//         ws.send(JSON.stringify({ type: 'UpdateConfiguration', keyterms: KEYTERMS, prompt: U3_PROMPT }));
//         return;
//       }

//       if (msg.type === 'Turn') {
//         const text: string = (msg.transcript || '').trim();
//         const aaiLang: string = msg.language_code ?? '';
//         const aaiConf: number = msg.language_confidence ?? 0;
//         const isFinal: boolean = msg.turn_is_formatted === true;
//         const wordCount = text.split(/\s+/).filter(Boolean).length;

//         const hasRealLang = !!(aaiLang && aaiLang !== 'undefined' && aaiConf > 0);
//         const isUniversalWord = /^(no|sí|si|yes|ok|yeah|bien|\d+)\.?,?$/i.test(text.trim());

//         if (isFinal && (s as any)._forceEndpointFallback) {
//           clearTimeout((s as any)._forceEndpointFallback);
//           delete (s as any)._forceEndpointFallback;
//         }

//         this.logger.log(`🔬 RAW fmt=${isFinal} lang=${aaiLang} conf=${aaiConf.toFixed(2)} "${text.substring(0, 60)}" [${sessionId}]`);
//         if (!text) return;

//         // ── Filtro de ruido: idioma no objetivo con baja confianza ────────────
//         const isNonTargetLang = hasRealLang && aaiLang !== 'en' && aaiLang !== 'es';
//         if (isNonTargetLang && aaiConf < 0.65 && wordCount <= 2 && !isUniversalWord) {
//           this.logger.log(`🚫 Ruido [${aaiLang}=${aaiConf.toFixed(2)}] "${text}" [${sessionId}]`);
//           return;
//         }

//         // ── Tipo A: artefactos de inicio de turno ────────────────────────────
//         if (isFinal && wordCount >= 2 && wordCount <= 4 && !buf.text) {
//           const tNorm = text.toLowerCase().replace(/[¿?!¡.,]/g, '').trim();
//           const knownArtifacts = [
//             'se despierta', 'qué hace como', 'es eso cómo', 'eso cómo',
//             'hace como', 'es eso', 'as you may know', 'señor', 'i see him',
//             'cómo hay', 'qué qué', 'qué es eso',
//           ];
//           const isKnownArtifact = knownArtifacts.some(a => tNorm === a);
//           const medicalOrCommon = /\b(convulsión|convulsiones|keppra|seizure|seizures|epilepsia|medicamento|dosis|dolor|espalda|cabeza|cerebro|hospital|doctor|médico|why|here|have|had|taking|your|you|when|last|how|many|what|were|before|after|increase|dose|missed|ever|day|days|sí|no|yes|okay|because|pero|desde|hace|tengo|tiene|tuve|dejé|pagar|cobrar|todos|días|años|meses|family|history|examine|examination|straight|follow|push|shoulder|extremities|strength|walking|pain|leg|back|run|down)\b/i;
//           const isGenericArtifact = wordCount <= 3 && text.endsWith('?') && !medicalOrCommon.test(text);
//           if (isKnownArtifact || isGenericArtifact) {
//             this.logger.log(`🗑️ Artefacto inicio [${sessionId}]: "${text}"`);
//             return;
//           }
//         }

//         // ── Tipo B: subtítulos/overlay o repetición ──────────────────────────
//         if (isFinal) {
//           const sentences = text.split(/[.!?¿¡]+/).map(s => s.trim()).filter(Boolean);
//           const normalized = sentences.map(s => s.toLowerCase().replace(/\s+/g, ' ').trim());
//           const hasDuplicateSentence = new Set(normalized).size < normalized.length && sentences.length >= 2;
//           const isMedicalList = sentences.length >= 3 && sentences.every(s => {
//             const words = s.trim().split(/\s+/).filter(Boolean);
//             return words.length <= 2 && /\b(seizure|seizures|epilepsy|epilepsia|convulsión|convulsiones|keppra|medication|medicamento)\b/i.test(s);
//           });
//           if (hasDuplicateSentence || isMedicalList) {
//             this.logger.log(`🗑️ Subtítulo/overlay [${sessionId}]: "${text.substring(0, 60)}"`);
//             return;
//           }
//         }

//         // ── Tipo C: fragmentos incompletos o ruido corto ─────────────────────
//         const endsWithDash = /[—–-]{1,2}$/.test(text.trim());
//         const hasTerminalPunct = /[.!?]$/.test(text.trim());
//         const isSingleMedicalWord = wordCount === 1 && /^(seizures?|epilepsy|epilepsia|convulsiones?|keppra|medication|medicamentos?)\.?$/i.test(text.trim());

//         if (isFinal && (endsWithDash || isSingleMedicalWord) && !buf.text) {
//           this.logger.log(`🗑️ Ruido suprimido [${sessionId}]: "${text}"`);
//           return;
//         }

//         const isIncompleteFragment = isFinal && !hasTerminalPunct && !endsWithDash && wordCount <= 4 && !buf.text;
//         if (isIncompleteFragment) {
//           this.logger.log(`⏳ Fragmento incompleto silencioso [${sessionId}]: "${text}"`);
//           buf.text = text;
//           buf.peakText = text;

//           // FIX: Detectar idioma correctamente ya en el fragmento inicial
//           const { lang: fragLex, strong: fragStrong } = this.detectLangWithStrength(text);
//           const startsObviouslyES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|y |no,|cuatro|tres|dos|uno)/i.test(text.trim());
//           if (fragStrong || startsObviouslyES) {
//             buf.lang = fragLex;
//             buf.langConfident = fragStrong;
//           } else {
//             buf.lang = this.resolveLang(text, aaiLang, aaiConf, buf.lastEmittedLang, wordCount);
//             buf.langConfident = hasRealLang;
//           }

//           buf.lastUpdateMs = now;
//           buf.lastSeenText = text;
//           buf.lastClosedMs = now;
//           this.clearTimer(buf);
//           buf.timer = setTimeout(() => {
//             buf.timer = null;
//             const sCheck = this.sessionData.get(sessionId);
//             if (sCheck?.buffer.text === text) {
//               this.logger.log(`🗑 Fragmento suprimido [${sessionId}]: "${text}"`);
//               if (sCheck) this.resetBuffer(sCheck.buffer);
//             }
//           }, 1400);
//           return;
//         }

//         // ── Corrección de idioma para texto español sin lang de AAI ──────────
//         // FIX: El log era solo informativo pero no cambiaba el idioma efectivamente.
//         // Ahora forzamos el idioma correcto ANTES de continuar.
//         const startsObviouslyES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|y |no,|cuatro|tres|dos|uno)/i.test(text.trim());
//         const { lang: preLex, strong: preStrong } = this.detectLangWithStrength(text);
//         const forceES = !hasRealLang && (preStrong && preLex === 'es') || startsObviouslyES;

//         if (isFinal && forceES && !buf.text) {
//           this.logger.log(`🔧 LangCorrect forzado ES [${sessionId}]: "${text.substring(0, 50)}"`);
//           buf.lang = 'es';
//           buf.langConfident = preStrong;
//         }

//         // ── Tipo D: filtro de intérprete ─────────────────────────────────────
//         if (isFinal) {
//           const tLow = text.toLowerCase().replace(/[¿?!¡.,]/g, '').trim();
//           const interpreterPatterns = [
//             /^cómo hay (muchos?|muchas?) convulsiones/,
//             /^así un medicamento/,
//             /^es antes o después de la dosis/,
//             /^tú tienes pastillas/,
//             /^qué fueron ustedes/,
//             /^estás tomando keppra/,
//             /^y hace cuánto tiempo tiene convulsiones/,
//             /^cuándo fue su últ[io]m[ao] convulsión/,
//             /^es eso lo que toma ahora/,
//             /^si alguna vez (ha|has|he) dejado de tomarla/,
//             /^hace cuánto tiempo tiene/,
//           ];
//           const isKnownInterpreter = interpreterPatterns.some(p => p.test(tLow));

//           const msSinceLastClose = now - buf.lastClosedMs;
//           const lastWasEn = buf.lastEmittedLang === 'en';
//           const isQuickEsAfterEn = lastWasEn && msSinceLastClose < 2000 && !buf.text;
//           const detectLangHere = this.resolveLang(text, aaiLang, aaiConf, buf.lastEmittedLang, wordCount);
//           let isSemanticInterpreter = false;
//           if (isQuickEsAfterEn && detectLangHere === 'es' && wordCount <= 10 && /[?]$/.test(text.trim())) {
//             const medTerms = /\b(convulsión|convulsiones|seizure|seizures|keppra|medicamento|medicamentos|dosis|dose|abril|april|junio|june|pastilla|pill|tomando|taking|dejó|stopped|cuánto|cuándo|when|antes|before|después|after|aumento|increase)\b/i;
//             const lastEnText = (buf.lastEmittedText || '').toLowerCase();
//             isSemanticInterpreter = medTerms.test(text) && medTerms.test(lastEnText);
//           }

//           if (isKnownInterpreter || isSemanticInterpreter) {
//             this.logger.log(`🎭 Intérprete filtrado [${sessionId}] (${isKnownInterpreter ? 'pattern' : 'semantic'} +${now - buf.lastClosedMs}ms): "${text}"`);
//             return;
//           }
//         }

//         // ── peakText solo crece con texto limpio ──────────────────────────────
//         const prevPeak = buf.peakText || '';
//         const isCleanGrowth = text.startsWith(prevPeak.substring(0, Math.min(prevPeak.length, 15)));
//         if (text.length > prevPeak.length && (isCleanGrowth || prevPeak.length < 10)) {
//           buf.peakText = text;
//         }

//         // ── Guard de continuación post-close ──────────────────────────────────
//         const msSinceClose = now - buf.lastClosedMs;
//         const msSinceForceClose = now - buf.forceClosedMs;

//         const normalize = (str: string) =>
//           str.replace(/Keppra/gi, 'kepra').replace(/[,\.!?¿¡—–]/g, '').replace(/\s+/g, ' ').trim().toLowerCase();

//         // Caso A: fragmento en buffer + llega texto completo → merge silencioso
//         if (buf.text && buf.text.split(/\s+/).length <= 4) {
//           const bufNorm = normalize(buf.text);
//           const newNorm = normalize(text);
//           if (newNorm.startsWith(bufNorm.substring(0, Math.min(bufNorm.length, 12))) && newNorm.length > bufNorm.length) {
//             this.logger.log(`🔁 FragmentMerge [${sessionId}]: "${buf.text}" → "${text.substring(0, 60)}"`);
//             this.clearTimer(buf);
//             buf.text = text;
//             buf.peakText = text;
//             buf.lastUpdateMs = now;
//             buf.lastSeenText = text;
//           }
//         }

//         // Caso B: turno cerrado recientemente y el nuevo texto lo extiende
//         if (!buf.text && msSinceClose < 1500 && buf.lastEmittedText && msSinceForceClose >= 2000) {
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

//         // Caso C: fragmento muy corto (1 palabra) de idioma diferente → flush primero
//         // Solo con 1 palabra porque con 2+ palabras puede ser una extensión legítima
//         // del mismo hablante (ej: buf="Sí." → nuevo="Sí, doctor." es el mismo Turn)
//         if (buf.text && buf.text.split(/\s+/).filter(Boolean).length === 1) {
//           const bufNorm = normalize(buf.text);
//           const newNorm = normalize(text);
//           const isExtension = newNorm.startsWith(bufNorm.substring(0, Math.min(bufNorm.length, 10)));
//           const isSameLang = buf.lang && this.resolveLang(text, aaiLang, aaiConf, buf.lang, wordCount) === buf.lang;
//           if (!isExtension && !isSameLang) {
//             this.logger.log(`🔀 FragmentFlush [${sessionId}]: "${buf.text}" → nuevo turno`);
//             const oldText = buf.text;
//             const oldLang = buf.lang;
//             this.resetBuffer(buf);
//             if (oldText) {
//               buf.text = oldText;
//               buf.lang = oldLang;
//               await this.closeTurn(sessionId, 'fragmentFlush');
//             }
//           }
//         }

//         const { lang: lexLang, strong: lexStrong } = this.detectLangWithStrength(text);
//         const silenceGap = now - buf.lastUpdateMs > 400;
//         const bufEmpty = !buf.lang || !buf.text;

//         // ── Asignación de idioma ──────────────────────────────────────────────
//         if (isUniversalWord && buf.lastEmittedLang) {
//           const isAmbiguousNo = /^no\.?,?$/i.test(text.trim());
//           const isDefinitelySpanish = /^(sí|sí,|si,)$/i.test(text.trim());
//           const isDefinitelyEnglish = /^(yes|yeah|nope)\.?,?$/i.test(text.trim());

//           if (isDefinitelySpanish) {
//             buf.lang = 'es';
//             buf.langConfident = false;
//           } else if (isDefinitelyEnglish) {
//             buf.lang = 'en';
//             buf.langConfident = false;
//           } else if (isAmbiguousNo) {
//             if (hasRealLang) {
//               buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
//               buf.langConfident = true;
//               this.logger.log(`🔄 AmbiguousNo→AAI [${buf.lang}] conf=${aaiConf.toFixed(2)} [${sessionId}]`);
//             } else {
//               const recentLangs = s.conversationHistory.slice(-2).map(h => h.lang);
//               const lastLang = recentLangs[recentLangs.length - 1] ?? buf.lastEmittedLang;
//               buf.lang = lastLang === 'en' ? 'es' : 'en';
//               buf.langConfident = false;
//               this.logger.log(`🔄 AmbiguousNo→History [${buf.lastEmittedLang}→${buf.lang}] [${sessionId}]`);
//             }
//           } else {
//             const isSpanishResponse = /^(sí|si)\.?,?$/i.test(text.trim());
//             if (isSpanishResponse && buf.lastEmittedLang === 'es') {
//               buf.lang = 'es';
//               buf.langConfident = false;
//             } else {
//               const opposite = buf.lastEmittedLang === 'en' ? 'es' : 'en';
//               buf.lang = opposite;
//               buf.langConfident = false;
//               this.logger.log(`🔄 UniversalFlip [${buf.lastEmittedLang}→${opposite}] "${text}" [${sessionId}]`);
//             }
//           }
//         } else if (bufEmpty && !buf.lang) {
//           // Solo asignar idioma si el buffer está completamente vacío (sin asignación previa del forceES)
//           if (hasRealLang) {
//             buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
//             buf.langConfident = true;
//             if (buf.lang !== buf.lastEmittedLang) {
//               this.logger.log(`🌍 LangFromAAI [${buf.lastEmittedLang}→${buf.lang}] [${sessionId}]`);
//             }
//           } else if (lexStrong && buf.lastEmittedLang && buf.lastEmittedLang !== lexLang) {
//             buf.lang = lexLang;
//             buf.langConfident = true;
//             this.logger.log(`🌍 LangFromLex [${buf.lastEmittedLang}→${lexLang}] [${sessionId}]`);
//           } else if (!lexStrong && this.isBackchannel(text) && buf.lastEmittedLang) {
//             const isSpanishBackchannel = /^(sí|si)\.?,?$/i.test(text.trim());
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
//         } else if (!bufEmpty && hasRealLang) {
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

//         // ── Speaker change ────────────────────────────────────────────────────
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

//         // ── Split de Turn mezclado EN + respuesta ES ──────────────────────────
//         if (isFinal && buf.lang === 'en') {
//           const mixedMatch = text.match(/^(.+\?)\s+((?:sí|si|no|claro|bien|okay|ok|cuatro|four|tres|three|dos|two|uno|one|\d+)[^?]*)$/i);
//           if (mixedMatch) {
//             const enPart = mixedMatch[1].trim();
//             const esPart = mixedMatch[2].trim();
//             const enWords = enPart.split(/\s+/).filter(Boolean);
//             const esWords = esPart.split(/\s+/).filter(Boolean).length;
//             const hasEnContent = enWords.some(w => /^(the|you|have|had|are|was|were|do|does|did|your|any|ever|since|before|after|how|when|what|why)$/i.test(w));
//             if (enWords.length >= 4 && esWords <= 6 && hasEnContent) {
//               this.logger.log(`✂️ Split EN+ES [${sessionId}]: EN="${enPart.substring(0, 50)}" ES="${esPart}"`);
//               buf.text = enPart;
//               buf.peakText = enPart;
//               buf.lang = 'en';
//               this.clearTimer(buf);
//               await this.closeTurn(sessionId, 'splitMixed');
//               const sAfterSplit = this.sessionData.get(sessionId);
//               if (sAfterSplit) {
//                 sAfterSplit.buffer.text = esPart;
//                 sAfterSplit.buffer.peakText = esPart;
//                 sAfterSplit.buffer.lang = 'es';
//                 sAfterSplit.buffer.langConfident = false;
//                 sAfterSplit.buffer.lastUpdateMs = now;
//                 await this.closeTurn(sessionId, 'splitMixed');
//               }
//               return;
//             }
//           }
//         }

//         // ── ForceClose por mezcla EN+ES ───────────────────────────────────────
//         if (wordCount >= 8 && buf.text) {
//           const words = text.trim().split(/\s+/);
//           const esOnly = /^(que|los|las|del|una|con|para|pero|desde|hace|porque|también|cuando|como|esto|eso|fue|han|tengo|tuve|tenía|convulsiones|días|mes|año|años|siempre|nunca|alguna|dejé|pagar|cobraba|incrementaron|tomarla|todos|ninguno|manejar|pregunta|exámenes|resultados|familia)$/i;
//           const enOnly = /^(the|and|you|have|had|are|taking|medications|seizures|since|before|after|dose|increase|missed|those|pills|times|every|medical|conditions|family|history|examine|when|was|your|last|seizure|not|examination|follow|straight|ahead|strength|walking|pain|leg|back)$/i;
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

//         // ── Silence timer ─────────────────────────────────────────────────────
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
//             // Dar más tiempo a turnos largos: el doctor a veces hace pausas mid-sentence
//             // en frases de 10+ palabras. Con poco tiempo se cierra antes de terminar.
//             const closeDelay = staleWords > 15 ? 1200
//               : staleWords > 8 ? 900
//               : T_SILENCE_CLOSE_STALE;
//             buf.timer = setTimeout(() => {
//               buf.timer = null;
//               const s2 = this.sessionData.get(sessionId);
//               if (s2?.ws?.readyState === 1 && s2.buffer.text) {
//                 this.logger.log(`⚡ ForceEndpoint [${sessionId}] (${staleWords}w)`);
//                 s2.ws.send(JSON.stringify({ type: 'ForceEndpoint' }));
//                 const fb = setTimeout(() => {
//                   this.logger.log(`⏱ Silence close fallback [${sessionId}]`);
//                   this.closeTurn(sessionId, 'silence');
//                 }, 800);
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
//       this.logger.log(`⏳ Cerrando AAI u3-rt-pro [${sessionId}]`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) await this.closeTurn(sessionId, 'userStop');
//       if (ws.readyState === WebSocket.OPEN) {
//         ws.send(JSON.stringify({ type: 'Terminate' }));
//         await new Promise((r) => setTimeout(r, 2500));
//       }
//       ws.close();
//       this.logger.log(`🛑 AAI u3-rt-pro cerrado [${sessionId}]`);
//     };

//     return { send, close };
//   }

//   private async claudePipeline(text: string, lang: 'es' | 'en', session: SessionData, sessionId: string) {
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
//       // Emitir siempre con el originalText para que el frontend pueda reemplazar
//       // el bloque correcto sin dejar el bloque viejo visible
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
//     text: string, lang: 'es' | 'en', _history: ConversationTurn[],
//   ): Promise<{ result: string; correctedLang: 'es' | 'en' }> {
//     // Claude desactivado — causaba eliminación de texto válido y correcciones incorrectas.
//     // fixText() ya maneja Keppra y 2,000. Solo aplicar correcciones 100% deterministas aquí.

//     let result = text;

//     // 1. Keppra (fixText ya lo hace, pero por si llega aquí antes)
//     result = result.replace(/\b(keprah?|kepra|quepra|kephra|kebri[ah]?|kebra)\b/gi, 'Keppra');

//     // 2. "Sí, 2000." → "Sí, 2,000." SOLO si el 2000 está solo como número de dosis
//     //    NO cambiar "2 pills", "2 times", "2 days" — solo el número puro 2000
//     result = result.replace(/\b2000\b/g, '2,000');

//     // 3. "si " al inicio → "Sí, " (solo si no tiene tilde ya)
//     result = result.replace(/^si\s/i, (m) => m[0] === 'S' ? 'Sí, ' : 'Sí, ');

//     if (this.norm(result) === this.norm(text)) return { result: text, correctedLang: lang };
//     const correctedLang: 'es' | 'en' = this.detectLang(result) ?? lang;
//     return { result, correctedLang };
//   }
// }
// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';
// import Anthropic from '@anthropic-ai/sdk';

// const T_SILENCE_CLOSE = 2500;        // 2.5s: paciente hace pausas naturales mid-sentence
// const T_SILENCE_CLOSE_STALE = 1200; // 1.2s: stale — mismo texto llegando, esperar más antes de ForceEndpoint
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
//     const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar|manejar|dolor|espalda|pregunta|exámenes|resultados|familia|ninguno)\b/g) || []).length;
//     const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes|examination|follow|straight|ahead|strength|walking)\b/g) || []).length;
//     return esScore > enScore ? 'es' : 'en';
//   }

//   private detectLangWithStrength(text: string): { lang: 'es' | 'en'; strong: boolean } {
//     const t = text.toLowerCase();
//     const esScore = (t.match(/\b(sí|si|de|el|la|los|las|por|para|que|en|me|te|se|nos|pero|desde|hace|porque|como|también|muy|bien|mal|ya|ahora|aquí|hospital|médico|medicina|convulsión|convulsiones|dejé|tomé|vine|volví|tengo|tiene|tuve|cerebro|años|meses|cuatro|tres|ninguno|pude|pagar|cobrar|manejar|dolor|espalda|pregunta|exámenes|resultados|familia|ninguno)\b/g) || []).length;
//     const enScore = (t.match(/\b(the|a|an|is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|can|i|you|he|she|we|they|my|your|his|her|and|or|but|if|when|where|why|how|what|which|who|that|this|here|now|before|after|doctor|patient|hospital|medication|seizure|seizures|keppra|dose|mg|gram|times|daily|four|three|none|no|yes|examination|follow|straight|ahead|strength|walking)\b/g) || []).length;
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

//     // FIX: No aplicar la regla de número → suprimir si el prefijo numérico
//     // es claramente una cantidad de pastillas (ej: "2 pills, 2 times a day")
//     // La regla original quitaba el "2" al inicio cuando el resto empezaba con palabra EN,
//     // lo que producía "pills, 2 times a day" — dejamos el número si va seguido de "pill/pills"
//     const enStartWords = /^(or|before|after|the|was|were|is|are|have|had|do|does|did|when|where|what|how|why|which|that|this|it|in|of|for|with|a|an|and|but|not|no|any|all|one|two|three|four|some|your|their|our|my|its)/i;
//     const shortPrefixMatch = t.match(/^(\d{1,3}\.?\s+)(\w.+)/);
//     if (shortPrefixMatch && enStartWords.test(shortPrefixMatch[2])) {
//       // Solo suprimir el número si lo que sigue NO es "pill(s)" o "tablet(s)"
//       const nextWord = shortPrefixMatch[2].split(/\s+/)[0].toLowerCase();
//       if (!/^pills?|tablets?$/.test(nextWord)) {
//         t = shortPrefixMatch[2].charAt(0).toUpperCase() + shortPrefixMatch[2].slice(1);
//       }
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

//     if (finalText.trim().length <= 2 && !isUniversalBackchannel && !isNumericResponse) {
//       this.logger.log(`🚫 Ruido corto [${sessionId}]: "${finalText}"`);
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
//     this.logger.log(`🎤 AssemblyAI u3-rt-pro iniciando [${sessionId}]`);

//     const params = new URLSearchParams({
//       sample_rate: '16000',
//       format_turns: 'true',
//       speech_model: 'u3-rt-pro',
//     });

//     const U3_PROMPT = 'Bilingual medical interpreter conversation. Doctor speaks English, patient speaks Spanish. Medical terminology includes seizures, Keppra, epilepsy, convulsiones, medicamentos. Do NOT translate — transcribe exactly as spoken in the original language.';

//     const KEYTERMS = [
//       'Keppra', 'convulsión', 'convulsiones', 'epilepsia',
//       'seizure', 'seizures', 'levetiracetam', 'medicamento',
//       'medicamentos', 'valproato', 'carbamazepina', 'lamotrigina',
//       'cerebro', 'dosis', 'electroencefalograma', 'MRI',
//     ];

//     const WebSocket = require('ws');
//     const ws = new WebSocket(
//       `wss://streaming.assemblyai.com/v3/ws?${params.toString()}`,
//       { headers: { Authorization: apiKey } },
//     );

//     session.ws = ws;
//     ws.on('open', () => this.logger.log(`✅ AssemblyAI u3-rt-pro abierto [${sessionId}]`));
//     ws.on('error', (err: Error) => this.logger.error(`❌ AAI error [${sessionId}]: ${err.message}`));

//     ws.on('close', (code: number) => {
//       this.logger.log(`🔒 AAI cerrado [${sessionId}] (${code})`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) this.closeTurn(sessionId, 'streamClose');
//       this.sessionData.delete(sessionId);
//     });

//     ws.on('message', async (raw: any) => {
//       const s = this.sessionData.get(sessionId);
//       if (!s) return;
//       let msg: any;
//       try { msg = JSON.parse(raw.toString()); } catch { return; }

//       const buf = s.buffer;
//       const now = Date.now();

//       if (msg.type === 'Begin') {
//         this.logger.log(`🔗 AAI u3-rt-pro [${sessionId}] sid=${msg.id}`);
//         ws.send(JSON.stringify({ type: 'UpdateConfiguration', keyterms: KEYTERMS, prompt: U3_PROMPT }));
//         return;
//       }

//       if (msg.type === 'Turn') {
//         const text: string = (msg.transcript || '').trim();
//         const aaiLang: string = msg.language_code ?? '';
//         const aaiConf: number = msg.language_confidence ?? 0;
//         const isFinal: boolean = msg.turn_is_formatted === true;
//         const wordCount = text.split(/\s+/).filter(Boolean).length;

//         const hasRealLang = !!(aaiLang && aaiLang !== 'undefined' && aaiConf > 0);
//         const isUniversalWord = /^(no|sí|si|yes|ok|yeah|bien|\d+)\.?,?$/i.test(text.trim());

//         if (isFinal && (s as any)._forceEndpointFallback) {
//           clearTimeout((s as any)._forceEndpointFallback);
//           delete (s as any)._forceEndpointFallback;
//         }

//         this.logger.log(`🔬 RAW fmt=${isFinal} lang=${aaiLang} conf=${aaiConf.toFixed(2)} "${text.substring(0, 60)}" [${sessionId}]`);
//         if (!text) return;

//         // ── Filtro de ruido: idioma no objetivo con baja confianza ────────────
//         const isNonTargetLang = hasRealLang && aaiLang !== 'en' && aaiLang !== 'es';
//         if (isNonTargetLang && aaiConf < 0.65 && wordCount <= 2 && !isUniversalWord) {
//           this.logger.log(`🚫 Ruido [${aaiLang}=${aaiConf.toFixed(2)}] "${text}" [${sessionId}]`);
//           return;
//         }

//         // ── Tipo A: artefactos de inicio de turno ────────────────────────────
//         if (isFinal && wordCount >= 2 && wordCount <= 4 && !buf.text) {
//           const tNorm = text.toLowerCase().replace(/[¿?!¡.,]/g, '').trim();
//           const knownArtifacts = [
//             'se despierta', 'qué hace como', 'es eso cómo', 'eso cómo',
//             'hace como', 'es eso', 'as you may know', 'señor', 'i see him',
//             'cómo hay', 'qué qué', 'qué es eso',
//           ];
//           const isKnownArtifact = knownArtifacts.some(a => tNorm === a);
//           const medicalOrCommon = /\b(convulsión|convulsiones|keppra|seizure|seizures|epilepsia|medicamento|dosis|dolor|espalda|cabeza|cerebro|hospital|doctor|médico|why|here|have|had|taking|your|you|when|last|how|many|what|were|before|after|increase|dose|missed|ever|day|days|sí|no|yes|okay|because|pero|desde|hace|tengo|tiene|tuve|dejé|pagar|cobrar|todos|días|años|meses|family|history|examine|examination|straight|follow|push|shoulder|extremities|strength|walking|pain|leg|back|run|down)\b/i;
//           const isGenericArtifact = wordCount <= 3 && text.endsWith('?') && !medicalOrCommon.test(text);
//           if (isKnownArtifact || isGenericArtifact) {
//             this.logger.log(`🗑️ Artefacto inicio [${sessionId}]: "${text}"`);
//             return;
//           }
//         }

//         // ── Tipo B: subtítulos/overlay o repetición ──────────────────────────
//         if (isFinal) {
//           const sentences = text.split(/[.!?¿¡]+/).map(s => s.trim()).filter(Boolean);
//           const normalized = sentences.map(s => s.toLowerCase().replace(/\s+/g, ' ').trim());
//           const hasDuplicateSentence = new Set(normalized).size < normalized.length && sentences.length >= 2;
//           const isMedicalList = sentences.length >= 3 && sentences.every(s => {
//             const words = s.trim().split(/\s+/).filter(Boolean);
//             return words.length <= 2 && /\b(seizure|seizures|epilepsy|epilepsia|convulsión|convulsiones|keppra|medication|medicamento)\b/i.test(s);
//           });
//           if (hasDuplicateSentence || isMedicalList) {
//             this.logger.log(`🗑️ Subtítulo/overlay [${sessionId}]: "${text.substring(0, 60)}"`);
//             return;
//           }
//         }

//         // ── Tipo C: fragmentos incompletos o ruido corto ─────────────────────
//         const endsWithDash = /[—–-]{1,2}$/.test(text.trim());
//         const hasTerminalPunct = /[.!?]$/.test(text.trim());
//         const isSingleMedicalWord = wordCount === 1 && /^(seizures?|epilepsy|epilepsia|convulsiones?|keppra|medication|medicamentos?)\.?$/i.test(text.trim());

//         if (isFinal && (endsWithDash || isSingleMedicalWord) && !buf.text) {
//           this.logger.log(`🗑️ Ruido suprimido [${sessionId}]: "${text}"`);
//           return;
//         }

//         const isIncompleteFragment = isFinal && !hasTerminalPunct && !endsWithDash && wordCount <= 4 && !buf.text;
//         if (isIncompleteFragment) {
//           this.logger.log(`⏳ Fragmento incompleto silencioso [${sessionId}]: "${text}"`);
//           buf.text = text;
//           buf.peakText = text;

//           // FIX: Detectar idioma correctamente ya en el fragmento inicial
//           const { lang: fragLex, strong: fragStrong } = this.detectLangWithStrength(text);
//           const startsObviouslyES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|y |no,|cuatro|tres|dos|uno)/i.test(text.trim());
//           if (fragStrong || startsObviouslyES) {
//             buf.lang = fragLex;
//             buf.langConfident = fragStrong;
//           } else {
//             buf.lang = this.resolveLang(text, aaiLang, aaiConf, buf.lastEmittedLang, wordCount);
//             buf.langConfident = hasRealLang;
//           }

//           buf.lastUpdateMs = now;
//           buf.lastSeenText = text;
//           buf.lastClosedMs = now;
//           this.clearTimer(buf);
//           buf.timer = setTimeout(() => {
//             buf.timer = null;
//             const sCheck = this.sessionData.get(sessionId);
//             if (sCheck?.buffer.text === text) {
//               this.logger.log(`🗑 Fragmento suprimido [${sessionId}]: "${text}"`);
//               if (sCheck) this.resetBuffer(sCheck.buffer);
//             }
//           }, 1400);
//           return;
//         }

//         // ── Corrección de idioma para texto español sin lang de AAI ──────────
//         // FIX: El log era solo informativo pero no cambiaba el idioma efectivamente.
//         // Ahora forzamos el idioma correcto ANTES de continuar.
//         const startsObviouslyES = /^(sí|si,|eso|hace|tengo|tiene|tuve|desde|pero|porque|y |no,|cuatro|tres|dos|uno)/i.test(text.trim());
//         const { lang: preLex, strong: preStrong } = this.detectLangWithStrength(text);
//         const forceES = !hasRealLang && (preStrong && preLex === 'es') || startsObviouslyES;

//         if (isFinal && forceES && !buf.text) {
//           this.logger.log(`🔧 LangCorrect forzado ES [${sessionId}]: "${text.substring(0, 50)}"`);
//           buf.lang = 'es';
//           buf.langConfident = preStrong;
//         }

//         // ── Tipo D: filtro de intérprete ─────────────────────────────────────
//         if (isFinal) {
//           const tLow = text.toLowerCase().replace(/[¿?!¡.,]/g, '').trim();
//           const interpreterPatterns = [
//             /^cómo hay (muchos?|muchas?) convulsiones/,
//             /^así un medicamento/,
//             /^es antes o después de la dosis/,
//             /^tú tienes pastillas/,
//             /^qué fueron ustedes/,
//             /^estás tomando keppra/,
//             /^y hace cuánto tiempo tiene convulsiones/,
//             /^cuándo fue su últ[io]m[ao] convulsión/,
//             /^es eso lo que toma ahora/,
//             /^si alguna vez (ha|has|he) dejado de tomarla/,
//             /^hace cuánto tiempo tiene/,
//           ];
//           const isKnownInterpreter = interpreterPatterns.some(p => p.test(tLow));

//           const msSinceLastClose = now - buf.lastClosedMs;
//           const lastWasEn = buf.lastEmittedLang === 'en';
//           const isQuickEsAfterEn = lastWasEn && msSinceLastClose < 2000 && !buf.text;
//           const detectLangHere = this.resolveLang(text, aaiLang, aaiConf, buf.lastEmittedLang, wordCount);
//           let isSemanticInterpreter = false;
//           if (isQuickEsAfterEn && detectLangHere === 'es' && wordCount <= 10 && /[?]$/.test(text.trim())) {
//             const medTerms = /\b(convulsión|convulsiones|seizure|seizures|keppra|medicamento|medicamentos|dosis|dose|abril|april|junio|june|pastilla|pill|tomando|taking|dejó|stopped|cuánto|cuándo|when|antes|before|después|after|aumento|increase)\b/i;
//             const lastEnText = (buf.lastEmittedText || '').toLowerCase();
//             isSemanticInterpreter = medTerms.test(text) && medTerms.test(lastEnText);
//           }

//           if (isKnownInterpreter || isSemanticInterpreter) {
//             this.logger.log(`🎭 Intérprete filtrado [${sessionId}] (${isKnownInterpreter ? 'pattern' : 'semantic'} +${now - buf.lastClosedMs}ms): "${text}"`);
//             return;
//           }
//         }

//         // ── peakText solo crece con texto limpio ──────────────────────────────
//         const prevPeak = buf.peakText || '';
//         const isCleanGrowth = text.startsWith(prevPeak.substring(0, Math.min(prevPeak.length, 15)));
//         if (text.length > prevPeak.length && (isCleanGrowth || prevPeak.length < 10)) {
//           buf.peakText = text;
//         }

//         // ── Guard de continuación post-close ──────────────────────────────────
//         const msSinceClose = now - buf.lastClosedMs;
//         const msSinceForceClose = now - buf.forceClosedMs;

//         const normalize = (str: string) =>
//           str.replace(/Keppra/gi, 'kepra').replace(/[,\.!?¿¡—–]/g, '').replace(/\s+/g, ' ').trim().toLowerCase();

//         // Caso A: fragmento en buffer + llega texto completo → merge silencioso
//         if (buf.text && buf.text.split(/\s+/).length <= 4) {
//           const bufNorm = normalize(buf.text);
//           const newNorm = normalize(text);
//           if (newNorm.startsWith(bufNorm.substring(0, Math.min(bufNorm.length, 12))) && newNorm.length > bufNorm.length) {
//             this.logger.log(`🔁 FragmentMerge [${sessionId}]: "${buf.text}" → "${text.substring(0, 60)}"`);
//             this.clearTimer(buf);
//             buf.text = text;
//             buf.peakText = text;
//             buf.lastUpdateMs = now;
//             buf.lastSeenText = text;
//           }
//         }

//         // Caso B: turno cerrado recientemente y el nuevo texto lo extiende
//         if (!buf.text && msSinceClose < 1800 && buf.lastEmittedText && msSinceForceClose >= 2000) {
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

//         // Caso C: fragmento muy corto (1 palabra) de idioma diferente → flush primero
//         // Solo con 1 palabra porque con 2+ palabras puede ser una extensión legítima
//         // del mismo hablante (ej: buf="Sí." → nuevo="Sí, doctor." es el mismo Turn)
//         if (buf.text && buf.text.split(/\s+/).filter(Boolean).length === 1) {
//           const bufNorm = normalize(buf.text);
//           const newNorm = normalize(text);
//           const isExtension = newNorm.startsWith(bufNorm.substring(0, Math.min(bufNorm.length, 10)));
//           const isSameLang = buf.lang && this.resolveLang(text, aaiLang, aaiConf, buf.lang, wordCount) === buf.lang;
//           if (!isExtension && !isSameLang) {
//             this.logger.log(`🔀 FragmentFlush [${sessionId}]: "${buf.text}" → nuevo turno`);
//             const oldText = buf.text;
//             const oldLang = buf.lang;
//             this.resetBuffer(buf);
//             if (oldText) {
//               buf.text = oldText;
//               buf.lang = oldLang;
//               await this.closeTurn(sessionId, 'fragmentFlush');
//             }
//           }
//         }

//         const { lang: lexLang, strong: lexStrong } = this.detectLangWithStrength(text);
//         const silenceGap = now - buf.lastUpdateMs > 400;
//         const bufEmpty = !buf.lang || !buf.text;

//         // ── Asignación de idioma ──────────────────────────────────────────────
//         if (isUniversalWord && buf.lastEmittedLang) {
//           const isAmbiguousNo = /^no\.?,?$/i.test(text.trim());
//           const isDefinitelySpanish = /^(sí|sí,|si,)$/i.test(text.trim());
//           const isDefinitelyEnglish = /^(yes|yeah|nope)\.?,?$/i.test(text.trim());

//           if (isDefinitelySpanish) {
//             buf.lang = 'es';
//             buf.langConfident = false;
//           } else if (isDefinitelyEnglish) {
//             buf.lang = 'en';
//             buf.langConfident = false;
//           } else if (isAmbiguousNo) {
//             if (hasRealLang) {
//               buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
//               buf.langConfident = true;
//               this.logger.log(`🔄 AmbiguousNo→AAI [${buf.lang}] conf=${aaiConf.toFixed(2)} [${sessionId}]`);
//             } else {
//               const recentLangs = s.conversationHistory.slice(-2).map(h => h.lang);
//               const lastLang = recentLangs[recentLangs.length - 1] ?? buf.lastEmittedLang;
//               buf.lang = lastLang === 'en' ? 'es' : 'en';
//               buf.langConfident = false;
//               this.logger.log(`🔄 AmbiguousNo→History [${buf.lastEmittedLang}→${buf.lang}] [${sessionId}]`);
//             }
//           } else {
//             const isSpanishResponse = /^(sí|si)\.?,?$/i.test(text.trim());
//             if (isSpanishResponse && buf.lastEmittedLang === 'es') {
//               buf.lang = 'es';
//               buf.langConfident = false;
//             } else {
//               const opposite = buf.lastEmittedLang === 'en' ? 'es' : 'en';
//               buf.lang = opposite;
//               buf.langConfident = false;
//               this.logger.log(`🔄 UniversalFlip [${buf.lastEmittedLang}→${opposite}] "${text}" [${sessionId}]`);
//             }
//           }
//         } else if (bufEmpty && !buf.lang) {
//           // Solo asignar idioma si el buffer está completamente vacío (sin asignación previa del forceES)
//           if (hasRealLang) {
//             buf.lang = aaiLang.startsWith('es') ? 'es' : 'en';
//             buf.langConfident = true;
//             if (buf.lang !== buf.lastEmittedLang) {
//               this.logger.log(`🌍 LangFromAAI [${buf.lastEmittedLang}→${buf.lang}] [${sessionId}]`);
//             }
//           } else if (lexStrong && buf.lastEmittedLang && buf.lastEmittedLang !== lexLang) {
//             buf.lang = lexLang;
//             buf.langConfident = true;
//             this.logger.log(`🌍 LangFromLex [${buf.lastEmittedLang}→${lexLang}] [${sessionId}]`);
//           } else if (!lexStrong && this.isBackchannel(text) && buf.lastEmittedLang) {
//             const isSpanishBackchannel = /^(sí|si)\.?,?$/i.test(text.trim());
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
//         } else if (!bufEmpty && hasRealLang) {
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

//         // ── Speaker change ────────────────────────────────────────────────────
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

//         // ── Split de Turn mezclado EN + respuesta ES ──────────────────────────
//         if (isFinal && buf.lang === 'en') {
//           const mixedMatch = text.match(/^(.+\?)\s+((?:sí|si|no|claro|bien|okay|ok|cuatro|four|tres|three|dos|two|uno|one|\d+)[^?]*)$/i);
//           if (mixedMatch) {
//             const enPart = mixedMatch[1].trim();
//             const esPart = mixedMatch[2].trim();
//             const enWords = enPart.split(/\s+/).filter(Boolean);
//             const esWords = esPart.split(/\s+/).filter(Boolean).length;
//             const hasEnContent = enWords.some(w => /^(the|you|have|had|are|was|were|do|does|did|your|any|ever|since|before|after|how|when|what|why)$/i.test(w));
//             if (enWords.length >= 4 && esWords <= 6 && hasEnContent) {
//               this.logger.log(`✂️ Split EN+ES [${sessionId}]: EN="${enPart.substring(0, 50)}" ES="${esPart}"`);
//               buf.text = enPart;
//               buf.peakText = enPart;
//               buf.lang = 'en';
//               this.clearTimer(buf);
//               await this.closeTurn(sessionId, 'splitMixed');
//               const sAfterSplit = this.sessionData.get(sessionId);
//               if (sAfterSplit) {
//                 sAfterSplit.buffer.text = esPart;
//                 sAfterSplit.buffer.peakText = esPart;
//                 sAfterSplit.buffer.lang = 'es';
//                 sAfterSplit.buffer.langConfident = false;
//                 sAfterSplit.buffer.lastUpdateMs = now;
//                 await this.closeTurn(sessionId, 'splitMixed');
//               }
//               return;
//             }
//           }
//         }

//         // ── ForceClose por mezcla EN+ES ───────────────────────────────────────
//         if (wordCount >= 8 && buf.text) {
//           const words = text.trim().split(/\s+/);
//           const esOnly = /^(que|los|las|del|una|con|para|pero|desde|hace|porque|también|cuando|como|esto|eso|fue|han|tengo|tuve|tenía|convulsiones|días|mes|año|años|siempre|nunca|alguna|dejé|pagar|cobraba|incrementaron|tomarla|todos|ninguno|manejar|pregunta|exámenes|resultados|familia)$/i;
//           const enOnly = /^(the|and|you|have|had|are|taking|medications|seizures|since|before|after|dose|increase|missed|those|pills|times|every|medical|conditions|family|history|examine|when|was|your|last|seizure|not|examination|follow|straight|ahead|strength|walking|pain|leg|back)$/i;
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

//         // ── Silence timer ─────────────────────────────────────────────────────
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
//             const closeDelay = staleWords > 12 ? 900 : T_SILENCE_CLOSE_STALE;
//             buf.timer = setTimeout(() => {
//               buf.timer = null;
//               const s2 = this.sessionData.get(sessionId);
//               if (s2?.ws?.readyState === 1 && s2.buffer.text) {
//                 this.logger.log(`⚡ ForceEndpoint [${sessionId}] (${staleWords}w)`);
//                 s2.ws.send(JSON.stringify({ type: 'ForceEndpoint' }));
//                 const fb = setTimeout(() => {
//                   this.logger.log(`⏱ Silence close fallback [${sessionId}]`);
//                   this.closeTurn(sessionId, 'silence');
//                 }, 800);
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
//       this.logger.log(`⏳ Cerrando AAI u3-rt-pro [${sessionId}]`);
//       const s = this.sessionData.get(sessionId);
//       if (s?.buffer.text || s?.buffer.peakText) await this.closeTurn(sessionId, 'userStop');
//       if (ws.readyState === WebSocket.OPEN) {
//         ws.send(JSON.stringify({ type: 'Terminate' }));
//         await new Promise((r) => setTimeout(r, 2500));
//       }
//       ws.close();
//       this.logger.log(`🛑 AAI u3-rt-pro cerrado [${sessionId}]`);
//     };

//     return { send, close };
//   }

//   private async claudePipeline(text: string, lang: 'es' | 'en', session: SessionData, sessionId: string) {
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
//       // Emitir siempre con el originalText para que el frontend pueda reemplazar
//       // el bloque correcto sin dejar el bloque viejo visible
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
//     text: string, lang: 'es' | 'en', _history: ConversationTurn[],
//   ): Promise<{ result: string; correctedLang: 'es' | 'en' }> {
//     // Claude desactivado — causaba eliminación de texto válido y correcciones incorrectas.
//     // fixText() ya maneja Keppra y 2,000. Solo aplicar correcciones 100% deterministas aquí.

//     let result = text;

//     // 1. Keppra (fixText ya lo hace, pero por si llega aquí antes)
//     result = result.replace(/\b(keprah?|kepra|quepra|kephra|kebri[ah]?|kebra)\b/gi, 'Keppra');

//     // 2. "Sí, 2000." → "Sí, 2,000." SOLO si el 2000 está solo como número de dosis
//     //    NO cambiar "2 pills", "2 times", "2 days" — solo el número puro 2000
//     result = result.replace(/\b2000\b/g, '2,000');

//     // 3. "si " al inicio → "Sí, " (solo si no tiene tilde ya)
//     result = result.replace(/^si\s/i, (m) => m[0] === 'S' ? 'Sí, ' : 'Sí, ');

//     if (this.norm(result) === this.norm(text)) return { result: text, correctedLang: lang };
//     const correctedLang: 'es' | 'en' = this.detectLang(result) ?? lang;
//     return { result, correctedLang };
//   }
// }