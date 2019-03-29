import { NgModule } from '@angular/core';
import { ThemeModule } from '../../@theme/theme.module';
import { RecordingsComponent } from './recordings.component';

@NgModule({
  imports: [
    ThemeModule
  ],
  declarations: [
    RecordingsComponent,
  ],
})
export class RecordingsModule { }
