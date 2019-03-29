import { NgModule } from '@angular/core';
import { ThemeModule } from '../../@theme/theme.module';
import { DashboardComponent } from './dashboard.component';
import { SocketIoModule, SocketIoConfig } from 'ngx-socket-io';
const config: SocketIoConfig = { url: 'http://localhost:5000', options: {} };

@NgModule({
  imports: [
    ThemeModule,
    SocketIoModule.forRoot(config)
  ],
  declarations: [
    DashboardComponent,
  ],
})
export class DashboardModule { }
