import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { TranscribeService } from './transcribe.service';

@Controller('transcribe')
export class TranscribeController {
  constructor(private readonly transcribeService: TranscribeService) {}

  @Post()
  @UseInterceptors(FileInterceptor('file'))
  async transcribe(@UploadedFile() file: Express.Multer.File) {
    return this.transcribeService.transcribe(file); // ← Ahora existe
  }
}

// import {
//   Controller,
//   Post,
//   UploadedFile,
//   UseInterceptors,
// } from '@nestjs/common';
// import { FileInterceptor } from '@nestjs/platform-express';
// import { TranscribeService } from './transcribe.service';

// @Controller('transcribe')
// export class TranscribeController {
//   constructor(private readonly transcribeService: TranscribeService) {}

//   @Post()
//   @UseInterceptors(FileInterceptor('file'))
//   async transcribe(@UploadedFile() file: Express.Multer.File) {
//     return this.transcribeService.transcribe(file);
//   }
// }
