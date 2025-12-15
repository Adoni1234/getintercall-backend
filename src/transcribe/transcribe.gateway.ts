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
    origin: [
      'http://localhost:4200',
      'https://localhost:4200',
      'getintercall-git-main-adonis-projects-9faf0f78.vercel.app',
      'https://getintercall.vercel.app',
    ],
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
    try {
      this.logger.log(`Llamando service para ${data.sessionId}`);
      const wsProxy = await this.transcribeService.startRealTimeTranscription(
        data.sessionId,
        (partialData: string) => {
          this.logger.log(
            `Emisión partial desde Gateway [${data.sessionId}]: ${partialData.substring(0, 50)}...`,
          );
          client.emit('partialTranscript', partialData); // String JSON
        },
      );
      client.data.ws = wsProxy;
      client.data.chunkQueue = []; // Queue for delayed chunks
      this.logger.log(`WS AssemblyAI asignado para ${data.sessionId}`);
      client.emit('started', {
        sessionId: data.sessionId,
        message: 'Real-time iniciado',
      }); // ← Emit after await
      // Flush queued chunks
      const queue = client.data.chunkQueue || [];
      for (const queuedChunk of queue) {
        wsProxy.send(queuedChunk);
        this.logger.log(
          `Queued chunk flushed for ${data.sessionId}: ${queuedChunk.byteLength} bytes`,
        );
      }
      delete client.data.chunkQueue;
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
    this.logger.log(
      `Recibido chunk para session ${data.sessionId}: ${chunkBuffer.byteLength} bytes desde ${client.id}`,
    );
    const wsProxy = client.data.ws;
    if (wsProxy && wsProxy.send) {
      wsProxy.send(chunkBuffer);
      this.logger.log(
        `Chunk enviado via proxy para ${data.sessionId}, reconstructed ${chunkBuffer.byteLength} bytes`,
      );
    } else {
      // Queue if not ready
      if (!client.data.chunkQueue) client.data.chunkQueue = [];
      client.data.chunkQueue.push(chunkBuffer);
      this.logger.warn(
        `Chunk queued for session ${data.sessionId} (proxy not ready yet)`,
      );
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
//     origin: ['http://localhost:4200', 'https://localhost:4200'], // ← Fix: Permite HTTP y HTTPS
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
//     client.emit('connected', { message: 'WS conectado!' });
//   }

//   @SubscribeMessage('startTranscription')
//   async handleStart(
//     @MessageBody() data: { sessionId: string },
//     @ConnectedSocket() client: Socket,
//   ) {
//     this.logger.log(
//       `Recibido startTranscription para session ${data.sessionId} desde ${client.id}`,
//     );
//     client.emit('started', {
//       sessionId: data.sessionId,
//       message: 'Real-time iniciado',
//     });
//     try {
//       this.logger.log(`Llamando service para ${data.sessionId}`);
//       const wsProxy = await this.transcribeService.startRealTimeTranscription(
//         data.sessionId,
//         (transcript) => {
//           this.logger.log(
//             `Emisión partial desde Gateway [${data.sessionId}]: ${transcript.substring(0, 50)}...`,
//           );
//           client.emit('partialTranscript', {
//             text: transcript,
//             sessionId: data.sessionId,
//           });
//         },
//       );
//       client.data.ws = wsProxy;
//       this.logger.log(`WS AssemblyAI asignado para ${data.sessionId}`);
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
//     const chunkArray = data.chunk; // Array from frontend
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
//       this.logger.warn(
//         `No WS proxy para session ${data.sessionId} – chunk descartado`,
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
//     }
//   }
// }

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
//     origin: 'http://localhost:4200',
//     methods: ['GET', 'POST'],
//     credentials: true,
//   },
// })
// export class TranscribeGateway implements OnGatewayConnection, OnGatewayDisconnect {
//   @WebSocketServer() server: Server;
//   private readonly logger = new Logger(TranscribeGateway.name);

//   handleConnection(client: Socket) {
//     this.logger.log(`Cliente conectado: ${client.id}`);
//     client.emit('connected', { message: 'WS conectado!' }); // ← Emit test
//   }

//   @SubscribeMessage('startTranscription')
//   async handleStart(@MessageBody() data: { sessionId: string }, @ConnectedSocket() client: Socket) {
//     this.logger.log(`Recibido startTranscription para session ${data.sessionId} desde ${client.id}`);
//     client.emit('started', { sessionId: data.sessionId, message: 'Real-time iniciado' });
//     // Aquí AssemblyAI real-time (comenta temporal para test conexión)
//     // const ws = await this.transcribeService.startRealTimeTranscription(data.sessionId, (transcript) => {
//     //   client.emit('partialTranscript', { text: transcript, sessionId: data.sessionId });
//     // });
//     // client.data.ws = ws;
//   }

//   @SubscribeMessage('audioChunk')
//   handleAudioChunk(@MessageBody() data: { sessionId: string; chunk: ArrayBuffer }, @ConnectedSocket() client: Socket) {
//     this.logger.log(`Recibido chunk para session ${data.sessionId}: ${data.chunk.byteLength} bytes desde ${client.id}`);
//     // Proxy a AssemblyAI (comenta temporal)
//     // if (client.data.ws) {
//     //   client.data.ws.send(data.chunk);
//     // }
//     client.emit('chunkReceived', { sessionId: data.sessionId, size: data.chunk.byteLength }); // ← Echo test
//   }

//   @SubscribeMessage('stopTranscription')
//   handleStop(@MessageBody() data: { sessionId: string }, @ConnectedSocket() client: Socket) {
//     this.logger.log(`Recibido stop para session ${data.sessionId} desde ${client.id}`);
//     client.emit('stopped', { sessionId: data.sessionId });
//   }

//   handleDisconnect(client: Socket) {
//     this.logger.log(`Cliente desconectado: ${client.id}`);
//   }
// }
