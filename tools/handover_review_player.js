/* Stored-video playback: a position is committed only with its decoded frame. */
(function (root) {
  'use strict';
  class ReviewPlayer {
    constructor(options) {
      this.loadFrame = options.loadFrame;
      this.commit = options.commit;
      this.dispose = options.dispose || (() => {});
      this.onChange = options.onChange || (() => {});
      this.onError = options.onError || (() => {});
      this.now = options.now || (() => performance.now());
      this.schedule = options.schedule || ((fn, ms) => setTimeout(fn, ms));
      this.unschedule = options.unschedule || (id => clearTimeout(id));
      this.session = null;
      this.times = [];
      this.index = null;
      this.requestedIndex = null;
      this.busy = false;
      this.playing = false;
      this.speed = 1;
      this.frameGeneration = 0;
      this.playGeneration = 0;
      this.timer = null;
      this.abort = null;
      this.requestKind = null;
    }
    get state() {
      return {session: this.session, index: this.index, requestedIndex: this.requestedIndex,
        busy: this.busy, playing: this.playing, speed: this.speed};
    }
    notify() { this.onChange(this.state); }
    cancelFrame() {
      this.frameGeneration += 1;
      if (this.abort) this.abort.abort();
      this.abort = null;
      this.busy = false;
      this.requestKind = null;
      this.requestedIndex = this.index;
    }
    pause() {
      this.playing = false;
      this.playGeneration += 1;
      if (this.timer !== null) this.unschedule(this.timer);
      this.timer = null;
      if (this.requestKind === 'play') this.cancelFrame();
      this.notify();
    }
    clear() {
      this.pause();
      this.cancelFrame();
      this.session = null;
      this.times = [];
      this.index = this.requestedIndex = null;
      this.notify();
    }
    async setSession(session, times) {
      this.clear();
      if (!session || !times.length || times.some((t, i) => !Number.isFinite(t) || (i && t <= times[i-1]))) {
        throw Error('Invalid recorded frame timeline');
      }
      this.session = session;
      this.times = times.slice();
      return this.request(0, 'seek');
    }
    async request(target, kind) {
      if (!this.session) return false;
      const index = Math.max(0, Math.min(this.times.length-1, Math.round(target)));
      const session = this.session;
      this.cancelFrame();
      const generation = this.frameGeneration;
      const abort = new AbortController();
      this.abort = abort;
      this.requestedIndex = index;
      this.busy = true;
      this.requestKind = kind;
      this.notify();
      const current = () => generation === this.frameGeneration && session === this.session;
      try {
        const resource = await this.loadFrame({session, index, signal: abort.signal});
        if (!current()) { this.dispose(resource); return false; }
        // The resource is already decoded. The DOM image and all position
        // metadata are committed in the same JavaScript turn by this callback.
        this.index = index;
        this.requestedIndex = index;
        this.busy = false;
        this.abort = null;
        this.requestKind = null;
        this.commit({session, index, resource});
        this.notify();
        return true;
      } catch (error) {
        if (!current() || error.name === 'AbortError') return false;
        this.busy = false;
        this.abort = null;
        this.requestKind = null;
        this.requestedIndex = this.index;
        this.playing = false;
        this.playGeneration += 1;
        this.notify();
        this.onError(error);
        return false;
      }
    }
    seek(index) { this.pause(); return this.request(index, 'seek'); }
    step(delta) {
      const from = this.busy ? this.requestedIndex : this.index;
      return this.seek((from === null ? 0 : from) + delta);
    }
    indexAt(time) {
      let lo = 0, hi = this.times.length;
      while (lo < hi) { const mid = (lo+hi) >> 1; if (this.times[mid] <= time) lo = mid+1; else hi = mid; }
      return Math.max(0, lo-1);
    }
    seekTime(time) { return this.seek(this.indexAt(time)); }
    play() {
      if (!this.session || this.index === null || this.busy) return false;
      this.pause();
      this.playing = true;
      this.originElapsed = this.times[this.index];
      this.originPerf = this.now();
      const generation = this.playGeneration;
      this.notify();
      this.timer = this.schedule(() => this.tick(generation), 80);
      return true;
    }
    toggle() { if (this.playing) this.pause(); else this.play(); }
    setSpeed(speed) {
      if (![.5, 1, 2].includes(speed)) throw Error('Unsupported playback speed');
      const wasPlaying = this.playing;
      this.pause();
      this.speed = speed;
      if (wasPlaying) this.play();
      else this.notify();
    }
    async tick(generation) {
      if (!this.playing || generation !== this.playGeneration) return;
      const target = this.originElapsed + (this.now()-this.originPerf)/1000*this.speed;
      const atEnd = target >= this.times[this.times.length-1];
      const index = atEnd ? this.times.length-1 : this.indexAt(target);
      if (index !== this.index) await this.request(index, 'play');
      // A pause, seek, speed change or session switch while decoding invalidates
      // this loop. It must never schedule another tick or overwrite a seek.
      if (!this.playing || generation !== this.playGeneration) return;
      if (atEnd) { this.pause(); return; }
      this.timer = this.schedule(() => this.tick(generation), 80);
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = ReviewPlayer;
  else root.ReviewPlayer = ReviewPlayer;
})(typeof window === 'undefined' ? globalThis : window);
