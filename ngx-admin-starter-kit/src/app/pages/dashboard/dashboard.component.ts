import { Component} from '@angular/core';
import { Socket } from 'ngx-socket-io';
import { HostListener } from '@angular/core'; 

@Component({
  selector: 'ngx-dashboard',
  styleUrls: ['./dashboard.component.css'],
  templateUrl: './dashboard.component.html',
})

export class DashboardComponent {

  constructor(private socket: Socket) {}

  ngOnInit() {
      var buttonRecord = document.getElementById("record") as HTMLInputElement;
      var buttonStop = document.getElementById("stop") as HTMLInputElement;

      buttonStop.disabled = true;

      buttonRecord.onclick = function() {
          buttonRecord.disabled = true;
          buttonStop.disabled = false;

          var xhr = new XMLHttpRequest();
          xhr.onreadystatechange = function() {
              if (xhr.readyState == 4 && xhr.status == 200) {}
          }
          xhr.open("POST", "http://localhost:5000/record_status", true);
          //xhr.withCredentials = true;
          xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
          xhr.send(JSON.stringify({status: "true"}));
      };

      buttonStop.onclick = function() {
          buttonRecord.disabled = false;
          buttonStop.disabled = true;

          var xhr = new XMLHttpRequest();
          xhr.onreadystatechange = function() {
              if (xhr.readyState == 4 && xhr.status == 200) {}
          }
          xhr.open("POST", "http://localhost:5000/record_status", true);
          //xhr.withCredentials = true;
          xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
          xhr.send(JSON.stringify({status: "false"}));
      };
  }

  move(msg: string){
    console.log(msg)
    this.socket.emit("move", msg);
  }

  @HostListener('document:keyup', ['$event'])
  handleDeleteKeyboardEvent(event: KeyboardEvent) {
    switch(event.key) {
      case "ArrowUp":
        this.move("forward");
        break
      case "ArrowDown":
        this.move("backward");
        break
      case "ArrowLeft":
        this.move("left");
        break
      case "ArrowRight":
        this.move("right");
        break
      case "/":
        this.move("stop");
        break
      case "w":
        this.move("forward");
        break
      case "s":
        this.move("backward");
        break
      case "d":
        this.move("right");
        break
      case "a":
        this.move("left");
        break
      case "f":
        this.move("stop");
        break
    }
  }

}
