import test from 'node:test';
import assert from 'node:assert/strict';
import ReviewPlayer from '../tools/handover_review_player.js';

function fixture() {
  const requests=[],commits=[],disposed=[],errors=[],timers=new Map();let timerId=0,clock=0;
  const player=new ReviewPlayer({
    loadFrame(request){return new Promise((resolve,reject)=>requests.push({...request,resolve,reject}));},
    commit(value){commits.push(value);},dispose:value=>disposed.push(value),onError:error=>errors.push(error),
    now:()=>clock,schedule:fn=>{timers.set(++timerId,fn);return timerId;},unschedule:id=>timers.delete(id)
  });
  async function init(){const p=player.setSession('a',[0,.1,.4,1.,2.]);requests[0].resolve('a0');await p;}
  function tick(time){clock=time;const [id,fn]=timers.entries().next().value;timers.delete(id);return fn();}
  return {player,requests,commits,disposed,errors,timers,init,tick};
}

test('the annotation index stays on the visible frame until decode completes',async()=>{
  const f=fixture();await f.init();const pending=f.player.seek(3);
  assert.equal(f.player.index,0);assert.equal(f.player.busy,true);assert.equal(f.commits.length,1);
  f.requests[1].resolve('a3');await pending;
  assert.equal(f.player.index,3);assert.equal(f.player.busy,false);assert.equal(f.commits.at(-1).resource,'a3');
});
test('late seek completion cannot overwrite a more recent seek',async()=>{
  const f=fixture();await f.init();const old=f.player.seek(1),current=f.player.seek(4);
  assert.equal(f.requests[1].signal.aborted,true);
  f.requests[2].resolve('a4');await current;f.requests[1].resolve('a1');await old;
  assert.equal(f.player.index,4);assert.equal(f.commits.at(-1).resource,'a4');assert.deepEqual(f.disposed,['a1']);
});
test('late frame from the old session cannot enter the new session',async()=>{
  const f=fixture();await f.init();const old=f.player.seek(2),replacement=f.player.setSession('b',[0,.3,.6]);
  f.requests[2].resolve('b0');await replacement;f.requests[1].resolve('a2');await old;
  assert.equal(f.player.session,'b');assert.equal(f.player.index,0);assert.equal(f.commits.at(-1).resource,'b0');
});
test('pause during decode freezes the displayed frame and cancels the old loop',async()=>{
  const f=fixture();await f.init();f.player.play();const running=f.tick(500);
  assert.equal(f.requests[1].index,2);f.player.pause();
  assert.equal(f.requests[1].signal.aborted,true);f.requests[1].resolve('late-play');await running;
  assert.equal(f.player.index,0);assert.equal(f.player.playing,false);assert.equal(f.timers.size,0);
});
test('pause, seek, restart cannot resurrect a second playback loop',async()=>{
  const f=fixture();await f.init();f.player.play();const old=f.tick(500);
  const seek=f.player.seek(3);f.requests[2].resolve('a3');await seek;f.player.play();
  assert.equal(f.timers.size,1);f.requests[1].resolve('old-play');await old;
  assert.equal(f.player.index,3);assert.equal(f.timers.size,1);assert.equal(f.player.playing,true);
});
test('playback follows irregular source timestamps rather than a nominal FPS',async()=>{
  const f=fixture();await f.init();f.player.play();const tick=f.tick(350);
  assert.equal(f.requests[1].index,1);assert.equal(f.player.index,0);
  f.requests[1].resolve('a1');await tick;
  assert.equal(f.player.index,1);assert.equal(f.timers.size,1);
});
test('failed image leaves a coherent old frame and stops playback',async()=>{
  const f=fixture();await f.init();f.player.play();const running=f.tick(500);
  f.requests[1].reject(Error('decode failed'));await running;
  assert.equal(f.player.index,0);assert.equal(f.player.busy,false);assert.equal(f.player.playing,false);
  assert.equal(f.timers.size,0);assert.equal(f.errors.length,1);
});
test('rapid next-frame clicks accumulate a target without advancing visible state early',async()=>{
  const f=fixture();await f.init();const a=f.player.step(1),b=f.player.step(1),c=f.player.step(1);
  assert.deepEqual(f.requests.slice(1).map(r=>r.index),[1,2,3]);assert.equal(f.player.index,0);
  f.requests[3].resolve('a3');await c;f.requests[1].resolve('a1');f.requests[2].resolve('a2');await a;await b;
  assert.equal(f.player.index,3);assert.equal(f.commits.length,2);
});
test('speed changes invalidate in-flight playback before restarting timing',async()=>{
  const f=fixture();await f.init();f.player.play();const old=f.tick(500);f.player.setSpeed(2);
  f.requests[1].resolve('old-speed');await old;
  assert.equal(f.player.index,0);assert.equal(f.player.speed,2);assert.equal(f.timers.size,1);
});

test('a recorded time gap holds the actual frame and next-frame can skip the wait',async()=>{
  const f=fixture();const init=f.player.setSession('gap',[0,.033,6.95,6.98]);
  f.requests[0].resolve('gap0');await init;
  const seek=f.player.seek(1);f.requests[1].resolve('gap1');await seek;
  f.player.play();await f.tick(1000);
  assert.equal(f.player.index,1);assert.equal(f.player.playing,true);
  assert.equal(f.requests.length,2); // No fabricated intermediate frames.
  const skip=f.player.step(1);f.requests[2].resolve('gap2');await skip;
  assert.equal(f.player.index,2);assert.equal(f.player.times[2],6.95);
  assert.equal(f.player.playing,false);assert.equal(f.timers.size,0);
});
