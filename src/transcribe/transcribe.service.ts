import { Injectable, Logger } from '@nestjs/common';
import { AssemblyAI } from 'assemblyai';

@Injectable()
export class TranscribeService {
  private readonly logger = new Logger(TranscribeService.name);
  private assembly: AssemblyAI | null;

  // 2000ms: más tolerante con pausas naturales de intérpretes
  private readonly END_SILENCE_THRESHOLD_MS = 2000;
  // Timer de seguridad: si AssemblyAI no envía is_final, lo forzamos
  private readonly FORCE_CLOSE_AFTER_MS = 2500;

  private sessionData = new Map<
    string,
    {
      accumulatedText: string;
      lastSentLength: number;
      chunkCount: number;
      turnTimer: NodeJS.Timeout | null;
      callback: (data: string) => void;
    }
  >();

  constructor() {
    const apiKey = process.env.ASSEMBLYAI_API_KEY;
    if (!apiKey) {
      this.logger.error('ASSEMBLYAI_API_KEY no encontrada – usando mock');
      this.assembly = null;
    } else {
      this.assembly = new AssemblyAI({ apiKey });
      this.logger.log('AssemblyAI real-time listo');
    }
  }

  async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
    try {
      this.logger.log(
        `Batch: ${file.originalname}, tamaño: ${file.size} bytes`,
      );
      if (!this.assembly) return { text: 'Mock batch: Audio procesado' };

      const transcript = await this.assembly.transcripts.transcribe({
        audio: file.buffer,
        language_code: 'es',
      });

      if (transcript.status === 'completed') {
        return { text: transcript.text || '' };
      } else {
        this.logger.warn(`Batch status: ${transcript.status}`);
        return { text: '' };
      }
    } catch (error) {
      this.logger.error(`Batch error: ${error.message}`);
      return { text: '' };
    }
  }

  private detectLanguage(text: string): 'es' | 'en' {
    const cleanText = text.toLowerCase().trim();

    if (/[áéíóúñ¿¡]/i.test(cleanText)) return 'es';

    const spanishGrammarPatterns = [
      /\b(que|qué)\s+(es|son|está|están|tiene|tienen)\b/i,
      /\b(el|la|los|las)\s+\w+\s+(de|del)\b/i,
      /\b(esto|esta|este|eso|esa|ese)\s+(es|son)\b/i,
      /\b(muy|más|menos)\s+\w+/i,
      /\b(no|si)\s+(puedo|puede|quiero|quiere|voy|va)\b/i,
      /\baquí\s+(es|está|en)\b/i,
      /\bestamos\s+(con|en)\b/i,
    ];
    if (spanishGrammarPatterns.some((p) => p.test(cleanText))) return 'es';

    const spanishPattern =
      /\b(de|del|el|la|los|las|un|una|está|están|son|es|como|qué|cómo|por|para|con|sin|pero|y|o|mi|tu|su|me|te|se|lo|le|ha|he|sido|sé|vamos|hacer|entonces|solo|mientras|lugares|más|nada|esto|no|que|muy|aquí|allí|allá|ahí|bien|mal|todo|siempre|nunca|cuando|donde|mucho|poco|grande|nuevo|bueno|malo|si|sí|ver|ir|voy|va|hago|dice|decir|ser|estar|tener|tengo|tiene|poder|puedo|querer|quiero|deber|debe|año|día|vez|cosa|gente|tiempo|vida|casa|ciudad|centro|desde|hasta|otro|mismo|cada|todos|estamos|sea)\b/gi;

    const words = cleanText.split(/\s+/).filter((w) => w.length > 0);
    const spanishMatches = cleanText.match(spanishPattern);
    const spanishWordCount = spanishMatches ? spanishMatches.length : 0;
    const spanishRatio = words.length > 0 ? spanishWordCount / words.length : 0;

    if (words.length <= 5 && spanishWordCount >= 1) return 'es';
    if (spanishRatio >= 0.18) return 'es';

    return 'en';
  }

  private forceCloseTurn(sessionId: string): void {
    const session = this.sessionData.get(sessionId);
    if (!session || !session.accumulatedText.trim()) return;

    const text = session.accumulatedText.trim();
    const lang = this.detectLanguage(text);

    this.logger.log(
      `⏱️ FORCE CLOSE [${sessionId}] [${lang}]: "${text.substring(0, 60)}"`,
    );

    session.callback(
      JSON.stringify({
        text,
        language: lang,
        isNewTurn: true,
        isForcedClose: true,
        sessionId,
      }),
    );

    session.accumulatedText = '';
    session.lastSentLength = 0;
    session.turnTimer = null;
  }

  private resetTurnTimer(sessionId: string): void {
    const session = this.sessionData.get(sessionId);
    if (!session) return;

    if (session.turnTimer) {
      clearTimeout(session.turnTimer);
      session.turnTimer = null;
    }

    if (session.accumulatedText.trim()) {
      session.turnTimer = setTimeout(() => {
        this.forceCloseTurn(sessionId);
      }, this.FORCE_CLOSE_AFTER_MS);
    }
  }

  async startRealTimeTranscription(
    sessionId: string,
    callback: (partialData: string) => void,
  ): Promise<any> {
    this.logger.log(`Iniciando real-time para session ${sessionId}`);

    if (!this.assembly) {
      const mockInterval = setInterval(() => {
        callback(
          JSON.stringify({
            text: `Mock [${sessionId}]: Hablando en vivo...`,
            language: 'es',
            isNewTurn: false,
            sessionId,
          }),
        );
      }, 2000);
      return { send: () => {}, close: () => clearInterval(mockInterval) };
    }

    this.sessionData.set(sessionId, {
      accumulatedText: '',
      lastSentLength: 0,
      chunkCount: 0,
      turnTimer: null,
      callback,
    });

    try {
      const config = {
        sampleRate: 16000,
        speechModel: 'universal-streaming-multilingual' as any,
        end_silence_threshold: this.END_SILENCE_THRESHOLD_MS,
        disable_partial_transcripts: false,
      };

      this.logger.log(
        `🎤 Config: sampleRate=16000, multilingual, silence=${this.END_SILENCE_THRESHOLD_MS}ms, forceClose=${this.FORCE_CLOSE_AFTER_MS}ms`,
      );

      const transcriber = this.assembly.streaming.transcriber(config);
      let isOpen = false;

      (transcriber.on as any)('open', (data: any) => {
        isOpen = true;
        this.logger.log(
          `✅ WS AssemblyAI abierto para ${sessionId} (ID: ${data.id})`,
        );
      });

      (transcriber.on as any)('turn', (data: any) => {
        const incomingText = (data.transcript || '').trim();
        const isFinal = data.is_final || false;

        if (!incomingText) return;

        const session = this.sessionData.get(sessionId);
        if (!session) return;

        const detectedLang = this.detectLanguage(incomingText);

        if (isFinal) {
          if (session.turnTimer) {
            clearTimeout(session.turnTimer);
            session.turnTimer = null;
          }

          this.logger.log(
            `✅ FINAL [${sessionId}] [${detectedLang}]: "${incomingText.substring(0, 80)}"`,
          );

          callback(
            JSON.stringify({
              text: incomingText,
              language: detectedLang,
              isNewTurn: true,
              sessionId,
            }),
          );

          session.accumulatedText = '';
          session.lastSentLength = 0;
        } else {
          // PARTIAL
          const isReformulation = incomingText.length < session.lastSentLength;

          if (isReformulation) {
            if (session.accumulatedText.trim()) {
              const prevLang = this.detectLanguage(session.accumulatedText);
              this.logger.log(
                `🔄 REFORMULACIÓN [${sessionId}]: cerrando bloque anterior`,
              );
              callback(
                JSON.stringify({
                  text: session.accumulatedText.trim(),
                  language: prevLang,
                  isNewTurn: true,
                  sessionId,
                }),
              );
            }

            session.accumulatedText = incomingText;
            session.lastSentLength = incomingText.length;

            callback(
              JSON.stringify({
                text: incomingText,
                language: detectedLang,
                isNewTurn: false,
                isNewBlock: true,
                sessionId,
              }),
            );
          } else if (incomingText.length > session.lastSentLength) {
            const newText = incomingText
              .substring(session.lastSentLength)
              .trim();
            session.accumulatedText = incomingText;
            session.lastSentLength = incomingText.length;

            if (newText) {
              this.logger.log(
                `📝 PARTIAL [${sessionId}] [${detectedLang}]: "+${newText.substring(0, 40)}"`,
              );
              callback(
                JSON.stringify({
                  text: newText,
                  language: detectedLang,
                  isNewTurn: false,
                  sessionId,
                }),
              );
            }
          }
          // Igual longitud → duplicado, ignorar

          this.resetTurnTimer(sessionId);
        }
      });

      transcriber.on('error', (error: any) => {
        this.logger.error(
          `❌ Error AssemblyAI [${sessionId}]: ${error.message}`,
        );
      });

      transcriber.on('close', (code: number, reason: string) => {
        this.logger.log(`WS cerrado [${sessionId}] (code: ${code})`);
        const session = this.sessionData.get(sessionId);
        if (session?.turnTimer) clearTimeout(session.turnTimer);
        this.sessionData.delete(sessionId);
        isOpen = false;
      });

      await transcriber.connect();
      this.logger.log(`✅ transcriber.connect() OK para ${sessionId}`);

      const sendChunk = (chunk: ArrayBuffer) => {
        if (!isOpen) {
          this.logger.warn(`⚠️ Chunk descartado [${sessionId}]: WS no abierto`);
          return;
        }
        const session = this.sessionData.get(sessionId);
        if (!session) return;

        session.chunkCount++;
        transcriber.sendAudio(chunk);

        if (session.chunkCount % 40 === 0) {
          const pcmData = new Int16Array(chunk);
          const avgLevel = this.calculateAudioLevel(pcmData);
          this.logger.log(
            `📤 [${sessionId}] Chunk #${session.chunkCount}: ${pcmData.length} samples, ${avgLevel.toFixed(1)}dB`,
          );
        }
      };

      return {
        send: sendChunk,
        close: () => {
          const session = this.sessionData.get(sessionId);
          if (session) {
            if (session.turnTimer) clearTimeout(session.turnTimer);
            if (session.accumulatedText.trim()) this.forceCloseTurn(sessionId);
          }
          transcriber.close();
        },
      };
    } catch (error) {
      this.logger.error(
        `❌ Error iniciando AssemblyAI [${sessionId}]: ${error.message}`,
      );
      this.sessionData.delete(sessionId);
      throw error; // Re-throw para que el gateway emita 'error' al cliente
    }
  }

  private calculateAudioLevel(buffer: Int16Array): number {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) sum += Math.abs(buffer[i]);
    const avg = sum / buffer.length;
    const normalized = avg / 32768;
    return 20 * Math.log10(normalized + 0.0001);
  }
}
// import { Injectable, Logger } from '@nestjs/common';
// import { AssemblyAI } from 'assemblyai';

// @Injectable()
// export class TranscribeService {
//   private readonly logger = new Logger(TranscribeService.name);
//   private assembly: AssemblyAI | null;

//   // Ajusta este valor según el ritmo de los intérpretes:
//   // 1000ms = corta rápido (bueno si hablan en frases cortas con pausa breve)
//   // 1500ms = balance recomendado
//   // 2000ms = más tolerante (bueno si hacen pausas naturales largas entre frases)
//   private readonly END_SILENCE_THRESHOLD_MS = 1500;

//   // Timer de seguridad en backend: si AssemblyAI no envía is_final, lo forzamos
//   // Debe ser mayor que END_SILENCE_THRESHOLD_MS
//   private readonly FORCE_CLOSE_AFTER_MS = 1800;

//   private sessionData = new Map<
//     string,
//     {
//       accumulatedText: string;
//       lastSentLength: number;
//       chunkCount: number;
//       turnTimer: NodeJS.Timeout | null;
//       callback: (data: string) => void;
//     }
//   >();

//   constructor() {
//     const apiKey = process.env.ASSEMBLYAI_API_KEY;
//     if (!apiKey) {
//       this.logger.error('ASSEMBLYAI_API_KEY no encontrada – usando mock');
//       this.assembly = null;
//     } else {
//       this.assembly = new AssemblyAI({ apiKey });
//       this.logger.log('AssemblyAI real-time listo');
//     }
//   }

//   async transcribe(file: Express.Multer.File): Promise<{ text: string }> {
//     try {
//       this.logger.log(
//         `Batch: ${file.originalname}, tamaño: ${file.size} bytes`,
//       );
//       if (!this.assembly) return { text: 'Mock batch: Audio procesado' };

//       const transcript = await this.assembly.transcripts.transcribe({
//         audio: file.buffer,
//         language_code: 'es',
//       });

//       if (transcript.status === 'completed') {
//         return { text: transcript.text || '' };
//       } else {
//         this.logger.warn(`Batch status: ${transcript.status}`);
//         return { text: '' };
//       }
//     } catch (error) {
//       this.logger.error(`Batch error: ${error.message}`);
//       return { text: '' };
//     }
//   }

//   private detectLanguage(text: string): 'es' | 'en' {
//     const cleanText = text.toLowerCase().trim();

//     if (/[áéíóúñ¿¡]/i.test(cleanText)) return 'es';

//     const spanishGrammarPatterns = [
//       /\b(que|qué)\s+(es|son|está|están|tiene|tienen)\b/i,
//       /\b(el|la|los|las)\s+\w+\s+(de|del)\b/i,
//       /\b(esto|esta|este|eso|esa|ese)\s+(es|son)\b/i,
//       /\b(muy|más|menos)\s+\w+/i,
//       /\b(no|si)\s+(puedo|puede|quiero|quiere|voy|va)\b/i,
//       /\baquí\s+(es|está|en)\b/i,
//       /\bestamos\s+(con|en)\b/i,
//     ];
//     if (spanishGrammarPatterns.some((p) => p.test(cleanText))) return 'es';

//     const spanishPattern =
//       /\b(de|del|el|la|los|las|un|una|está|están|son|es|como|qué|cómo|por|para|con|sin|pero|y|o|mi|tu|su|me|te|se|lo|le|ha|he|sido|sé|vamos|hacer|entonces|solo|mientras|lugares|más|nada|esto|no|que|muy|aquí|allí|allá|ahí|bien|mal|todo|siempre|nunca|cuando|donde|mucho|poco|grande|nuevo|bueno|malo|si|sí|ver|ir|voy|va|hago|dice|decir|ser|estar|tener|tengo|tiene|poder|puedo|querer|quiero|deber|debe|año|día|vez|cosa|gente|tiempo|vida|casa|ciudad|centro|desde|hasta|otro|mismo|cada|todos|estamos|sea)\b/gi;

//     const words = cleanText.split(/\s+/).filter((w) => w.length > 0);
//     const spanishMatches = cleanText.match(spanishPattern);
//     const spanishWordCount = spanishMatches ? spanishMatches.length : 0;
//     const spanishRatio = words.length > 0 ? spanishWordCount / words.length : 0;

//     if (words.length <= 5 && spanishWordCount >= 1) return 'es';
//     if (spanishRatio >= 0.18) return 'es';

//     return 'en';
//   }

//   private forceCloseTurn(sessionId: string): void {
//     const session = this.sessionData.get(sessionId);
//     if (!session || !session.accumulatedText.trim()) return;

//     const text = session.accumulatedText.trim();
//     const lang = this.detectLanguage(text);

//     this.logger.log(
//       `⏱️ FORCE CLOSE TURN [${sessionId}] [${lang}]: "${text.substring(0, 60)}"`,
//     );

//     session.callback(
//       JSON.stringify({
//         text,
//         language: lang,
//         isNewTurn: true,
//         isForcedClose: true,
//         sessionId,
//       }),
//     );

//     session.accumulatedText = '';
//     session.lastSentLength = 0;
//     session.turnTimer = null;
//   }

//   private resetTurnTimer(sessionId: string): void {
//     const session = this.sessionData.get(sessionId);
//     if (!session) return;

//     if (session.turnTimer) {
//       clearTimeout(session.turnTimer);
//       session.turnTimer = null;
//     }

//     if (session.accumulatedText.trim()) {
//       session.turnTimer = setTimeout(() => {
//         this.forceCloseTurn(sessionId);
//       }, this.FORCE_CLOSE_AFTER_MS);
//     }
//   }

//   async startRealTimeTranscription(
//     sessionId: string,
//     callback: (partialData: string) => void,
//   ): Promise<any> {
//     this.logger.log(`Iniciando real-time para session ${sessionId}`);

//     if (!this.assembly) {
//       this.logger.log(`Mock real-time para ${sessionId}`);
//       const mockInterval = setInterval(() => {
//         callback(
//           JSON.stringify({
//             text: `Mock [${sessionId}]: Hablando en vivo...`,
//             language: 'es',
//             isNewTurn: false,
//             sessionId,
//           }),
//         );
//       }, 2000);
//       return { send: () => {}, close: () => clearInterval(mockInterval) };
//     }

//     this.sessionData.set(sessionId, {
//       accumulatedText: '',
//       lastSentLength: 0,
//       chunkCount: 0,
//       turnTimer: null,
//       callback,
//     });

//     try {
//       const config = {
//         sampleRate: 16000,
//         speechModel: 'universal-streaming-multilingual' as any,
//         end_silence_threshold: this.END_SILENCE_THRESHOLD_MS,
//         disable_partial_transcripts: false,
//       };

//       this.logger.log(
//         `🎤 Config: sampleRate=16000, modelo=multilingual, ` +
//           `silence=${this.END_SILENCE_THRESHOLD_MS}ms, forceClose=${this.FORCE_CLOSE_AFTER_MS}ms`,
//       );

//       const transcriber = this.assembly.streaming.transcriber(config);
//       let isOpen = false;

//       (transcriber.on as any)('open', (data: any) => {
//         isOpen = true;
//         this.logger.log(
//           `✅ WS AssemblyAI abierto para ${sessionId} (ID: ${data.id})`,
//         );
//       });

//       (transcriber.on as any)('turn', (data: any) => {
//         const incomingText = (data.transcript || '').trim();
//         const isFinal = data.is_final || false;

//         if (!incomingText) return;

//         const session = this.sessionData.get(sessionId);
//         if (!session) return;

//         const detectedLang = this.detectLanguage(incomingText);

//         if (isFinal) {
//           // AssemblyAI cerró el turno — cancelar nuestro timer
//           if (session.turnTimer) {
//             clearTimeout(session.turnTimer);
//             session.turnTimer = null;
//           }

//           this.logger.log(
//             `✅ FINAL [${sessionId}] [${detectedLang}]: "${incomingText.substring(0, 80)}"`,
//           );

//           callback(
//             JSON.stringify({
//               text: incomingText,
//               language: detectedLang,
//               isNewTurn: true,
//               sessionId,
//             }),
//           );

//           session.accumulatedText = '';
//           session.lastSentLength = 0;
//         } else {
//           // PARTIAL
//           const isReformulation = incomingText.length < session.lastSentLength;

//           if (isReformulation) {
//             // AssemblyAI reinició su contexto — cerrar bloque anterior y abrir nuevo
//             if (session.accumulatedText.trim()) {
//               const prevLang = this.detectLanguage(session.accumulatedText);
//               this.logger.log(
//                 `🔄 REFORMULACIÓN → cierre forzado [${sessionId}]: "${session.accumulatedText.substring(0, 60)}"`,
//               );
//               callback(
//                 JSON.stringify({
//                   text: session.accumulatedText.trim(),
//                   language: prevLang,
//                   isNewTurn: true,
//                   sessionId,
//                 }),
//               );
//             }

//             session.accumulatedText = incomingText;
//             session.lastSentLength = incomingText.length;

//             this.logger.log(
//               `🆕 NUEVO BLOQUE tras reformulación [${sessionId}] [${detectedLang}]: "${incomingText.substring(0, 60)}"`,
//             );

//             callback(
//               JSON.stringify({
//                 text: incomingText,
//                 language: detectedLang,
//                 isNewTurn: false,
//                 isNewBlock: true,
//                 sessionId,
//               }),
//             );
//           } else if (incomingText.length > session.lastSentLength) {
//             // Hay texto nuevo
//             const newText = incomingText
//               .substring(session.lastSentLength)
//               .trim();

//             session.accumulatedText = incomingText;
//             session.lastSentLength = incomingText.length;

//             if (newText) {
//               this.logger.log(
//                 `📝 PARTIAL [${sessionId}] [${detectedLang}]: "+${newText.substring(0, 40)}" (total: ${incomingText.length})`,
//               );
//               callback(
//                 JSON.stringify({
//                   text: newText,
//                   language: detectedLang,
//                   isNewTurn: false,
//                   sessionId,
//                 }),
//               );
//             }
//           }
//           // Longitud igual → duplicado, ignorar

//           // Reiniciar timer de cierre forzado
//           this.resetTurnTimer(sessionId);
//         }
//       });

//       transcriber.on('error', (error: any) => {
//         this.logger.error(
//           `❌ Error AssemblyAI [${sessionId}]: ${error.message}`,
//         );
//       });

//       transcriber.on('close', (code: number, reason: string) => {
//         this.logger.log(
//           `WS cerrado [${sessionId}] (code: ${code}, reason: ${reason})`,
//         );
//         const session = this.sessionData.get(sessionId);
//         if (session?.turnTimer) clearTimeout(session.turnTimer);
//         this.sessionData.delete(sessionId);
//         isOpen = false;
//       });

//       await transcriber.connect();
//       this.logger.log(`transcriber.connect() OK para ${sessionId}`);

//       const sendChunk = (chunk: ArrayBuffer) => {
//         if (!isOpen) {
//           this.logger.warn(`⚠️ Chunk descartado [${sessionId}]: WS no abierto`);
//           return;
//         }
//         const session = this.sessionData.get(sessionId);
//         if (!session) return;

//         session.chunkCount++;
//         transcriber.sendAudio(chunk);

//         if (session.chunkCount % 40 === 0) {
//           const pcmData = new Int16Array(chunk);
//           const avgLevel = this.calculateAudioLevel(pcmData);
//           this.logger.log(
//             `📤 [${sessionId}] Chunk #${session.chunkCount}: ${pcmData.length} samples, nivel: ${avgLevel.toFixed(1)}dB`,
//           );
//         }
//       };

//       return {
//         send: sendChunk,
//         close: () => {
//           const session = this.sessionData.get(sessionId);
//           if (session) {
//             if (session.turnTimer) clearTimeout(session.turnTimer);
//             if (session.accumulatedText.trim()) this.forceCloseTurn(sessionId);
//           }
//           transcriber.close();
//         },
//       };
//     } catch (error) {
//       this.logger.error(
//         `❌ Error iniciando AssemblyAI [${sessionId}]: ${error.message}`,
//       );
//       const fallbackInterval = setInterval(() => {
//         callback(
//           JSON.stringify({
//             text: 'Error conectando con AssemblyAI',
//             language: 'es',
//             isNewTurn: true,
//             sessionId,
//           }),
//         );
//       }, 5000);
//       return { send: () => {}, close: () => clearInterval(fallbackInterval) };
//     }
//   }

//   private calculateAudioLevel(buffer: Int16Array): number {
//     let sum = 0;
//     for (let i = 0; i < buffer.length; i++) sum += Math.abs(buffer[i]);
//     const avg = sum / buffer.length;
//     const normalized = avg / 32768;
//     return 20 * Math.log10(normalized + 0.0001);
//   }
// }
