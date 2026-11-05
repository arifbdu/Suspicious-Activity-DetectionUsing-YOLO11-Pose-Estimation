const form=document.getElementById('upload-form');
const btn=document.getElementById('upload-btn');const statusEl=document.getElementById('status');const prog=document.getElementById('progress');
const live=document.getElementById('live-preview');
const err=document.getElementById('error');const video=document.getElementById('result-video');const blank=document.getElementById('placeholder');
const summary=document.getElementById('summary');const download=document.getElementById('download');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function text(el,value){el.textContent=value;el.hidden=false;}
form.addEventListener('submit',async function(event){
  event.preventDefault();err.hidden=true;live.hidden=true;live.removeAttribute('src');video.hidden=true;video.removeAttribute('src');video.load();
  download.hidden=true;summary.hidden=true;blank.hidden=false;prog.value=0;prog.hidden=false;
  btn.disabled=true;btn.textContent='Processing Video...';text(statusEl,'Uploading video…');
  try{
    const response=await fetch(form.action,{method:'POST',body:new FormData(form)});
    let data=await response.json().catch(()=>({error:'Server returned an invalid response.'}));
    if(!response.ok)throw new Error(data.error||`Upload error HTTP ${response.status}`);
    live.src=data.stream_url+'?t='+Date.now();live.hidden=false;blank.hidden=true;
    text(statusEl,'Live detection starting…');
    while(true){
      await delay(1200);
      const poll=await fetch(data.status_url,{cache:'no-store'});
      const job=await poll.json();
      if(!poll.ok)throw new Error(job.error||'Could not retrieve job status.');
      text(statusEl,job.message||'Analyzing…');
      if(job.percentage!=null)prog.value=job.percentage;
      else prog.removeAttribute('value');
      if(job.status==='error')throw new Error(job.message||'Processing failed.');
      if(job.status==='done'){
        prog.value=100;prog.hidden=true;statusEl.hidden=true;
        const result=job.result;
        live.hidden=true;live.removeAttribute('src');
        video.src=result.video;video.poster=result.preview;video.hidden=false;blank.hidden=true;
        text(summary,`${result.analyzed_frames} of ${result.frames} frames analyzed · ${result.flagged_frames} analyzed frames flagged · Up to ${result.max_people} people detected · ${result.device.toUpperCase()} inference`);
        download.href=result.download;download.hidden=false;
        video.load();
        video.onerror=()=>text(err,'This browser could not decode the processed video. Try Chrome or Edge and check that H.264 encoding completed.');
        // Autoplay with audio is restricted by browsers. Muting allows inline playback;
        // viewers can unmute using the video controls.
        video.muted=true;
        video.play().catch(()=>{text(statusEl,'Video ready. Press Play to watch in this page.');});
        break;
      }
    }
  }catch(e){live.hidden=true;live.removeAttribute('src');text(err,e.message);prog.hidden=true;text(statusEl,'Processing unsuccessful. See error above and terminal logs.');}
  finally{btn.disabled=false;btn.textContent='Upload & Start Detection';}
});
