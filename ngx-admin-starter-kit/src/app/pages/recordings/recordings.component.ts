import { Component, Injectable } from '@angular/core';

@Component({
  selector: 'ngx-recordings',
  styleUrls: ['./recordings.component.css'],
  templateUrl: './recordings.component.html',
})

export class RecordingsComponent {
    lol = new Array();
    ngOnInit() {
        var oReq = new XMLHttpRequest();
        oReq.open("GET", "http://localhost:5000/recordings");
        oReq.onload = () => {
            if (oReq.status === 200) {
                this.lol = JSON.parse(oReq.responseText).num_videos;
                console.log(this.lol);
                this.lol = Array(this.lol);
            }
            else {
                alert('Request failed.  Returned status of ' + oReq.status);
            }
        };
        oReq.send();
    }

    viewRecording(recording: number) {
        var video_player = document.getElementById('video-player') as HTMLVideoElement;
        video_player.setAttribute('src', 'http://localhost:5000/recording?recording_no='+recording);
        video_player.load();
    }
    onSelect(recording: number) {
        console.log('lol');
        var video_player = document.getElementById('video-player') as HTMLVideoElement;
        video_player.setAttribute('src', 'http://localhost:5000/recording?recording_no='+recording);
        video_player.load();
    }
}
