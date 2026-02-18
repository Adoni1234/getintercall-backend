import {
  WebSocketGateway,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
  WebSocketServer,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Logger } from '@nestjs/common';
import { TranscribeService } from './transcribe.service';

@WebSocketGateway({
  cors: {
    origin: ['http://localhost:4200', 'https://localhost:4200'],
    methods: ['GET', 'POST'],
    credentials: true,
  },
})
export class TranscribeGateway
  implements OnGatewayConnection, OnGatewayDisconnect
{
  @WebSocketServer() server: Server;
  private readonly logger = new Logger(TranscribeGateway.name);

  constructor(private transcribeService: TranscribeService) {}

  handleConnection(client: Socket) {
    this.logger.log(`Cliente conectado: ${client.id}`);
  }

  @SubscribeMessage('startTranscription')
  async handleStart(
    @MessageBody() data: { sessionId: string },
    @ConnectedSocket() client: Socket,
  ) {
    this.logger.log(
      `Recibido startTranscription para session ${data.sessionId} desde ${client.id}`,
    );

    // 🔥 IMPORTANTE: Inicializar queue ANTES de llamar al service
    client.data.chunkQueue = [];

    try {
      this.logger.log(`Llamando service para ${data.sessionId}`);
      const wsProxy = await this.transcribeService.startRealTimeTranscription(
        data.sessionId,
        (partialData: string) => {
          this.logger.log(
            `Emisión partial desde Gateway [${data.sessionId}]: ${partialData.substring(0, 50)}...`,
          );
          client.emit('partialTranscript', partialData);
        },
      );

      // 🔥 Asignar wsProxy ANTES de emitir 'started'
      client.data.ws = wsProxy;

      this.logger.log(`WS AssemblyAI asignado para ${data.sessionId}`);

      // 🔥 CRITICAL: Flush queued chunks ANTES de emitir 'started'
      const queue = client.data.chunkQueue || [];
      this.logger.log(
        `🔥 Flushing ${queue.length} queued chunks para ${data.sessionId}`,
      );

      for (const queuedChunk of queue) {
        wsProxy.send(queuedChunk);
      }

      if (queue.length > 0) {
        this.logger.log(
          `✅ ${queue.length} chunks flushed para ${data.sessionId}`,
        );
      }

      // 🔥 Limpiar queue después de flush
      client.data.chunkQueue = [];

      // 🔥 AHORA SÍ emitir 'started' - frontend empezará a enviar chunks
      client.emit('started', {
        sessionId: data.sessionId,
        message: 'Real-time iniciado',
      });
    } catch (error) {
      this.logger.error(
        `Error iniciando AssemblyAI para ${data.sessionId}: ${error.message}`,
      );
      client.emit('error', { message: error.message });
    }
  }

  @SubscribeMessage('audioChunk')
  handleAudioChunk(
    @MessageBody() data: { sessionId: string; chunk: number[] },
    @ConnectedSocket() client: Socket,
  ) {
    const chunkArray = data.chunk;
    const chunkBuffer = new Uint8Array(chunkArray).buffer;

    const wsProxy = client.data.ws;
    if (wsProxy && wsProxy.send) {
      // 🔥 WS está listo, enviar directamente
      wsProxy.send(chunkBuffer);

      // Log reducido (cada 40 chunks = 1 segundo)
      const count = (client.data.chunkCount || 0) + 1;
      client.data.chunkCount = count;

      if (count % 40 === 0) {
        this.logger.log(
          `📤 Chunk #${count} enviado para ${data.sessionId}: ${chunkBuffer.byteLength} bytes`,
        );
      }
    } else {
      // 🔥 WS aún no listo, guardar en queue
      if (!client.data.chunkQueue) client.data.chunkQueue = [];
      client.data.chunkQueue.push(chunkBuffer);

      if (client.data.chunkQueue.length % 10 === 0) {
        this.logger.warn(
          `⏳ ${client.data.chunkQueue.length} chunks queued para ${data.sessionId} (esperando WS)`,
        );
      }
    }

    client.emit('chunkReceived', {
      sessionId: data.sessionId,
      size: chunkBuffer.byteLength,
    });
  }

  @SubscribeMessage('stopTranscription')
  handleStop(
    @MessageBody() data: { sessionId: string },
    @ConnectedSocket() client: Socket,
  ) {
    this.logger.log(
      `Recibido stop para session ${data.sessionId} desde ${client.id}`,
    );
    const wsProxy = client.data.ws;
    if (wsProxy && wsProxy.close) {
      wsProxy.close();
      client.data.ws = null;
      delete client.data.chunkQueue;
      delete client.data.chunkCount;
      this.logger.log(`WS AssemblyAI cerrado para ${data.sessionId}`);
    }
    client.emit('stopped', { sessionId: data.sessionId });
  }

  handleDisconnect(client: Socket) {
    this.logger.log(`Cliente desconectado: ${client.id}`);
    const wsProxy = client.data.ws;
    if (wsProxy && wsProxy.close) {
      wsProxy.close();
      client.data.ws = null;
      delete client.data.chunkQueue;
      delete client.data.chunkCount;
    }
  }
}

// import {
//   WebSocketGateway,
//   SubscribeMessage,
//   MessageBody,
//   ConnectedSocket,
//   WebSocketServer,
//   OnGatewayConnection,
//   OnGatewayDisconnect,
// } from '@nestjs/websockets';
// import { Server, Socket } from 'socket.io';
// import { Logger } from '@nestjs/common';
// import { TranscribeService } from './transcribe.service';

// @WebSocketGateway({
//   cors: {
//     origin: [
//       'http://localhost:4200',
//       'https://localhost:4200',
//       'getintercall-git-main-adonis-projects-9faf0f78.vercel.app',
//       'https://getintercall.vercel.app',
//     ],
//     methods: ['GET', 'POST'],
//     credentials: true,
//   },
// })
// export class TranscribeGateway
//   implements OnGatewayConnection, OnGatewayDisconnect
// {
//   @WebSocketServer() server: Server;
//   private readonly logger = new Logger(TranscribeGateway.name);

//   constructor(private transcribeService: TranscribeService) {}

//   handleConnection(client: Socket) {
//     this.logger.log(`Cliente conectado: ${client.id}`);
//   }

//   @SubscribeMessage('startTranscription')
//   async handleStart(
//     @MessageBody() data: { sessionId: string },
//     @ConnectedSocket() client: Socket,
//   ) {
//     this.logger.log(
//       `Recibido startTranscription para session ${data.sessionId} desde ${client.id}`,
//     );
//     try {
//       this.logger.log(`Llamando service para ${data.sessionId}`);
//       const wsProxy = await this.transcribeService.startRealTimeTranscription(
//         data.sessionId,
//         (partialData: string) => {
//           this.logger.log(
//             `Emisión partial desde Gateway [${data.sessionId}]: ${partialData.substring(0, 50)}...`,
//           );
//           client.emit('partialTranscript', partialData); // String JSON
//         },
//       );
//       client.data.ws = wsProxy;
//       client.data.chunkQueue = []; // Queue for delayed chunks
//       this.logger.log(`WS AssemblyAI asignado para ${data.sessionId}`);
//       client.emit('started', {
//         sessionId: data.sessionId,
//         message: 'Real-time iniciado',
//       }); // ← Emit after await
//       // Flush queued chunks
//       const queue = client.data.chunkQueue || [];
//       for (const queuedChunk of queue) {
//         wsProxy.send(queuedChunk);
//         this.logger.log(
//           `Queued chunk flushed for ${data.sessionId}: ${queuedChunk.byteLength} bytes`,
//         );
//       }
//       delete client.data.chunkQueue;
//     } catch (error) {
//       this.logger.error(
//         `Error iniciando AssemblyAI para ${data.sessionId}: ${error.message}`,
//       );
//       client.emit('error', { message: error.message });
//     }
//   }

//   @SubscribeMessage('audioChunk')
//   handleAudioChunk(
//     @MessageBody() data: { sessionId: string; chunk: number[] },
//     @ConnectedSocket() client: Socket,
//   ) {
//     const chunkArray = data.chunk;
//     const chunkBuffer = new Uint8Array(chunkArray).buffer;
//     this.logger.log(
//       `Recibido chunk para session ${data.sessionId}: ${chunkBuffer.byteLength} bytes desde ${client.id}`,
//     );
//     const wsProxy = client.data.ws;
//     if (wsProxy && wsProxy.send) {
//       wsProxy.send(chunkBuffer);
//       this.logger.log(
//         `Chunk enviado via proxy para ${data.sessionId}, reconstructed ${chunkBuffer.byteLength} bytes`,
//       );
//     } else {
//       // Queue if not ready
//       if (!client.data.chunkQueue) client.data.chunkQueue = [];
//       client.data.chunkQueue.push(chunkBuffer);
//       this.logger.warn(
//         `Chunk queued for session ${data.sessionId} (proxy not ready yet)`,
//       );
//     }
//     client.emit('chunkReceived', {
//       sessionId: data.sessionId,
//       size: chunkBuffer.byteLength,
//     });
//   }

//   @SubscribeMessage('stopTranscription')
//   handleStop(
//     @MessageBody() data: { sessionId: string },
//     @ConnectedSocket() client: Socket,
//   ) {
//     this.logger.log(
//       `Recibido stop para session ${data.sessionId} desde ${client.id}`,
//     );
//     const wsProxy = client.data.ws;
//     if (wsProxy && wsProxy.close) {
//       wsProxy.close();
//       client.data.ws = null;
//       delete client.data.chunkQueue;
//       this.logger.log(`WS AssemblyAI cerrado para ${data.sessionId}`);
//     }
//     client.emit('stopped', { sessionId: data.sessionId });
//   }

//   handleDisconnect(client: Socket) {
//     this.logger.log(`Cliente desconectado: ${client.id}`);
//     const wsProxy = client.data.ws;
//     if (wsProxy && wsProxy.close) {
//       wsProxy.close();
//       client.data.ws = null;
//       delete client.data.chunkQueue;
//     }
//   }
// }
