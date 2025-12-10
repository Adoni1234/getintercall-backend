import { Module } from '@nestjs/common';
import { TranscribeController } from './transcribe.controller';
import { TranscribeService } from './transcribe.service';
import { TranscribeGateway } from './transcribe.gateway';

@Module({
  controllers: [TranscribeController],
  providers: [TranscribeService, TranscribeGateway], // ← Gateway como provider
  exports: [TranscribeGateway], // ← Exporta para AppModule
})
export class TranscribeModule {}
