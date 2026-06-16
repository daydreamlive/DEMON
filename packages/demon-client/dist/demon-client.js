// types/wireContract.gen.ts
var PROTOCOL_VERSION = 1;
var KNOB_SCHEMA_VERSION = 1;
var COMMAND_NAMES = [
  "params",
  "loop_band",
  "prompt",
  "set_prompt_blend",
  "set_interp_method",
  "set_depth",
  "enable_lora",
  "disable_lora",
  "manual_slot_add",
  "manual_slot_pop",
  "set_timbre_strength",
  "set_timbre_source",
  "set_timbre_fixture",
  "clear_timbre_source",
  "set_structure_source",
  "set_structure_fixture",
  "clear_structure_source",
  "swap_source",
  "write_audio"
];
var EVENT_NAMES = [
  "init_ack",
  "ready",
  "error",
  "params_update",
  "params_echo",
  "prompt_blend_echo",
  "prompt_applied",
  "lora_catalog",
  "swap_ready",
  "swap_failed",
  "stem_assets",
  "stem_failed",
  "depth_applied",
  "manual_slot_count",
  "timbre_set",
  "timbre_cleared",
  "timbre_failed",
  "structure_set",
  "structure_cleared",
  "structure_failed",
  "audio_written",
  "audio_write_failed",
  "command_failed"
];

// types/protocol.ts
var SAMPLE_RATE = 48e3;
var T = 1500;
var CROSSFADE_SECONDS = 0.025;
var SLICE_HDR_SIZE = 23;
var SLICE_FLAG_RAW = 0;
var SLICE_FLAG_DELTA = 1;
var PREEMPTED_CLOSE_CODE = 4001;

// node_modules/fzstd/esm/index.mjs
var ab = ArrayBuffer;
var u8 = Uint8Array;
var u16 = Uint16Array;
var i16 = Int16Array;
var i32 = Int32Array;
var slc = function(v, s, e) {
  if (u8.prototype.slice)
    return u8.prototype.slice.call(v, s, e);
  if (s == null || s < 0)
    s = 0;
  if (e == null || e > v.length)
    e = v.length;
  var n = new u8(e - s);
  n.set(v.subarray(s, e));
  return n;
};
var fill = function(v, n, s, e) {
  if (u8.prototype.fill)
    return u8.prototype.fill.call(v, n, s, e);
  if (s == null || s < 0)
    s = 0;
  if (e == null || e > v.length)
    e = v.length;
  for (; s < e; ++s)
    v[s] = n;
  return v;
};
var cpw = function(v, t, s, e) {
  if (u8.prototype.copyWithin)
    return u8.prototype.copyWithin.call(v, t, s, e);
  if (s == null || s < 0)
    s = 0;
  if (e == null || e > v.length)
    e = v.length;
  while (s < e) {
    v[t++] = v[s++];
  }
};
var ec = [
  "invalid zstd data",
  "window size too large (>2046MB)",
  "invalid block type",
  "FSE accuracy too high",
  "match distance too far back",
  "unexpected EOF"
];
var err = function(ind, msg, nt) {
  var e = new Error(msg || ec[ind]);
  e.code = ind;
  if (Error.captureStackTrace)
    Error.captureStackTrace(e, err);
  if (!nt)
    throw e;
  return e;
};
var rb = function(d, b, n) {
  var i = 0, o = 0;
  for (; i < n; ++i)
    o |= d[b++] << (i << 3);
  return o;
};
var b4 = function(d, b) {
  return (d[b] | d[b + 1] << 8 | d[b + 2] << 16 | d[b + 3] << 24) >>> 0;
};
var rzfh = function(dat, w) {
  var n3 = dat[0] | dat[1] << 8 | dat[2] << 16;
  if (n3 == 3126568 && dat[3] == 253) {
    var flg = dat[4];
    var ss = flg >> 5 & 1, cc = flg >> 2 & 1, df = flg & 3, fcf = flg >> 6;
    if (flg & 8)
      err(0);
    var bt = 6 - ss;
    var db = df == 3 ? 4 : df;
    var di = rb(dat, bt, db);
    bt += db;
    var fsb = fcf ? 1 << fcf : ss;
    var fss = rb(dat, bt, fsb) + (fcf == 1 && 256);
    var ws = fss;
    if (!ss) {
      var wb = 1 << 10 + (dat[5] >> 3);
      ws = wb + (wb >> 3) * (dat[5] & 7);
    }
    if (ws > 2145386496)
      err(1);
    var buf = new u8((w == 1 ? fss || ws : w ? 0 : ws) + 12);
    buf[0] = 1, buf[4] = 4, buf[8] = 8;
    return {
      b: bt + fsb,
      y: 0,
      l: 0,
      d: di,
      w: w && w != 1 ? w : buf.subarray(12),
      e: ws,
      o: new i32(buf.buffer, 0, 3),
      u: fss,
      c: cc,
      m: Math.min(131072, ws)
    };
  } else if ((n3 >> 4 | dat[3] << 20) == 25481893) {
    return b4(dat, 4) + 8;
  }
  err(0);
};
var msb = function(val) {
  var bits = 0;
  for (; 1 << bits <= val; ++bits)
    ;
  return bits - 1;
};
var rfse = function(dat, bt, mal) {
  var tpos = (bt << 3) + 4;
  var al = (dat[bt] & 15) + 5;
  if (al > mal)
    err(3);
  var sz = 1 << al;
  var probs = sz, sym = -1, re = -1, i = -1, ht = sz;
  var buf = new ab(512 + (sz << 2));
  var freq = new i16(buf, 0, 256);
  var dstate = new u16(buf, 0, 256);
  var nstate = new u16(buf, 512, sz);
  var bb1 = 512 + (sz << 1);
  var syms = new u8(buf, bb1, sz);
  var nbits = new u8(buf, bb1 + sz);
  while (sym < 255 && probs > 0) {
    var bits = msb(probs + 1);
    var cbt = tpos >> 3;
    var msk = (1 << bits + 1) - 1;
    var val = (dat[cbt] | dat[cbt + 1] << 8 | dat[cbt + 2] << 16) >> (tpos & 7) & msk;
    var msk1fb = (1 << bits) - 1;
    var msv = msk - probs - 1;
    var sval = val & msk1fb;
    if (sval < msv)
      tpos += bits, val = sval;
    else {
      tpos += bits + 1;
      if (val > msk1fb)
        val -= msv;
    }
    freq[++sym] = --val;
    if (val == -1) {
      probs += val;
      syms[--ht] = sym;
    } else
      probs -= val;
    if (!val) {
      do {
        var rbt = tpos >> 3;
        re = (dat[rbt] | dat[rbt + 1] << 8) >> (tpos & 7) & 3;
        tpos += 2;
        sym += re;
      } while (re == 3);
    }
  }
  if (sym > 255 || probs)
    err(0);
  var sympos = 0;
  var sstep = (sz >> 1) + (sz >> 3) + 3;
  var smask = sz - 1;
  for (var s = 0; s <= sym; ++s) {
    var sf = freq[s];
    if (sf < 1) {
      dstate[s] = -sf;
      continue;
    }
    for (i = 0; i < sf; ++i) {
      syms[sympos] = s;
      do {
        sympos = sympos + sstep & smask;
      } while (sympos >= ht);
    }
  }
  if (sympos)
    err(0);
  for (i = 0; i < sz; ++i) {
    var ns = dstate[syms[i]]++;
    var nb = nbits[i] = al - msb(ns);
    nstate[i] = (ns << nb) - sz;
  }
  return [tpos + 7 >> 3, {
    b: al,
    s: syms,
    n: nbits,
    t: nstate
  }];
};
var rhu = function(dat, bt) {
  var i = 0, wc = -1;
  var buf = new u8(292), hb = dat[bt];
  var hw = buf.subarray(0, 256);
  var rc = buf.subarray(256, 268);
  var ri = new u16(buf.buffer, 268);
  if (hb < 128) {
    var _a = rfse(dat, bt + 1, 6), ebt = _a[0], fdt = _a[1];
    bt += hb;
    var epos = ebt << 3;
    var lb = dat[bt];
    if (!lb)
      err(0);
    var st1 = 0, st2 = 0, btr1 = fdt.b, btr2 = btr1;
    var fpos = (++bt << 3) - 8 + msb(lb);
    for (; ; ) {
      fpos -= btr1;
      if (fpos < epos)
        break;
      var cbt = fpos >> 3;
      st1 += (dat[cbt] | dat[cbt + 1] << 8) >> (fpos & 7) & (1 << btr1) - 1;
      hw[++wc] = fdt.s[st1];
      fpos -= btr2;
      if (fpos < epos)
        break;
      cbt = fpos >> 3;
      st2 += (dat[cbt] | dat[cbt + 1] << 8) >> (fpos & 7) & (1 << btr2) - 1;
      hw[++wc] = fdt.s[st2];
      btr1 = fdt.n[st1];
      st1 = fdt.t[st1];
      btr2 = fdt.n[st2];
      st2 = fdt.t[st2];
    }
    if (++wc > 255)
      err(0);
  } else {
    wc = hb - 127;
    for (; i < wc; i += 2) {
      var byte = dat[++bt];
      hw[i] = byte >> 4;
      hw[i + 1] = byte & 15;
    }
    ++bt;
  }
  var wes = 0;
  for (i = 0; i < wc; ++i) {
    var wt = hw[i];
    if (wt > 11)
      err(0);
    wes += wt && 1 << wt - 1;
  }
  var mb = msb(wes) + 1;
  var ts = 1 << mb;
  var rem = ts - wes;
  if (rem & rem - 1)
    err(0);
  hw[wc++] = msb(rem) + 1;
  for (i = 0; i < wc; ++i) {
    var wt = hw[i];
    ++rc[hw[i] = wt && mb + 1 - wt];
  }
  var hbuf = new u8(ts << 1);
  var syms = hbuf.subarray(0, ts), nb = hbuf.subarray(ts);
  ri[mb] = 0;
  for (i = mb; i > 0; --i) {
    var pv = ri[i];
    fill(nb, i, pv, ri[i - 1] = pv + rc[i] * (1 << mb - i));
  }
  if (ri[0] != ts)
    err(0);
  for (i = 0; i < wc; ++i) {
    var bits = hw[i];
    if (bits) {
      var code = ri[bits];
      fill(syms, i, code, ri[bits] = code + (1 << mb - bits));
    }
  }
  return [bt, {
    n: nb,
    b: mb,
    s: syms
  }];
};
var dllt = rfse(/* @__PURE__ */ new u8([
  81,
  16,
  99,
  140,
  49,
  198,
  24,
  99,
  12,
  33,
  196,
  24,
  99,
  102,
  102,
  134,
  70,
  146,
  4
]), 0, 6)[1];
var dmlt = rfse(/* @__PURE__ */ new u8([
  33,
  20,
  196,
  24,
  99,
  140,
  33,
  132,
  16,
  66,
  8,
  33,
  132,
  16,
  66,
  8,
  33,
  68,
  68,
  68,
  68,
  68,
  68,
  68,
  68,
  36,
  9
]), 0, 6)[1];
var doct = rfse(/* @__PURE__ */ new u8([
  32,
  132,
  16,
  66,
  102,
  70,
  68,
  68,
  68,
  68,
  36,
  73,
  2
]), 0, 5)[1];
var b2bl = function(b, s) {
  var len = b.length, bl = new i32(len);
  for (var i = 0; i < len; ++i) {
    bl[i] = s;
    s += 1 << b[i];
  }
  return bl;
};
var llb = /* @__PURE__ */ new u8((/* @__PURE__ */ new i32([
  0,
  0,
  0,
  0,
  16843009,
  50528770,
  134678020,
  202050057,
  269422093
])).buffer, 0, 36);
var llbl = /* @__PURE__ */ b2bl(llb, 0);
var mlb = /* @__PURE__ */ new u8((/* @__PURE__ */ new i32([
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  16843009,
  50528770,
  117769220,
  185207048,
  252579084,
  16
])).buffer, 0, 53);
var mlbl = /* @__PURE__ */ b2bl(mlb, 3);
var dhu = function(dat, out, hu) {
  var len = dat.length, ss = out.length, lb = dat[len - 1], msk = (1 << hu.b) - 1, eb = -hu.b;
  if (!lb)
    err(0);
  var st = 0, btr = hu.b, pos = (len << 3) - 8 + msb(lb) - btr, i = -1;
  for (; pos > eb && i < ss; ) {
    var cbt = pos >> 3;
    var val = (dat[cbt] | dat[cbt + 1] << 8 | dat[cbt + 2] << 16) >> (pos & 7);
    st = (st << btr | val) & msk;
    out[++i] = hu.s[st];
    pos -= btr = hu.n[st];
  }
  if (pos != eb || i + 1 != ss)
    err(0);
};
var dhu4 = function(dat, out, hu) {
  var bt = 6;
  var ss = out.length, sz1 = ss + 3 >> 2, sz2 = sz1 << 1, sz3 = sz1 + sz2;
  dhu(dat.subarray(bt, bt += dat[0] | dat[1] << 8), out.subarray(0, sz1), hu);
  dhu(dat.subarray(bt, bt += dat[2] | dat[3] << 8), out.subarray(sz1, sz2), hu);
  dhu(dat.subarray(bt, bt += dat[4] | dat[5] << 8), out.subarray(sz2, sz3), hu);
  dhu(dat.subarray(bt), out.subarray(sz3), hu);
};
var rzb = function(dat, st, out) {
  var _a;
  var bt = st.b;
  var b0 = dat[bt], btype = b0 >> 1 & 3;
  st.l = b0 & 1;
  var sz = b0 >> 3 | dat[bt + 1] << 5 | dat[bt + 2] << 13;
  var ebt = (bt += 3) + sz;
  if (btype == 1) {
    if (bt >= dat.length)
      return;
    st.b = bt + 1;
    if (out) {
      fill(out, dat[bt], st.y, st.y += sz);
      return out;
    }
    return fill(new u8(sz), dat[bt]);
  }
  if (ebt > dat.length)
    return;
  if (btype == 0) {
    st.b = ebt;
    if (out) {
      out.set(dat.subarray(bt, ebt), st.y);
      st.y += sz;
      return out;
    }
    return slc(dat, bt, ebt);
  }
  if (btype == 2) {
    var b3 = dat[bt], lbt = b3 & 3, sf = b3 >> 2 & 3;
    var lss = b3 >> 4, lcs = 0, s4 = 0;
    if (lbt < 2) {
      if (sf & 1)
        lss |= dat[++bt] << 4 | (sf & 2 && dat[++bt] << 12);
      else
        lss = b3 >> 3;
    } else {
      s4 = sf;
      if (sf < 2)
        lss |= (dat[++bt] & 63) << 4, lcs = dat[bt] >> 6 | dat[++bt] << 2;
      else if (sf == 2)
        lss |= dat[++bt] << 4 | (dat[++bt] & 3) << 12, lcs = dat[bt] >> 2 | dat[++bt] << 6;
      else
        lss |= dat[++bt] << 4 | (dat[++bt] & 63) << 12, lcs = dat[bt] >> 6 | dat[++bt] << 2 | dat[++bt] << 10;
    }
    ++bt;
    var buf = out ? out.subarray(st.y, st.y + st.m) : new u8(st.m);
    var spl = buf.length - lss;
    if (lbt == 0)
      buf.set(dat.subarray(bt, bt += lss), spl);
    else if (lbt == 1)
      fill(buf, dat[bt++], spl);
    else {
      var hu = st.h;
      if (lbt == 2) {
        var hud = rhu(dat, bt);
        lcs += bt - (bt = hud[0]);
        st.h = hu = hud[1];
      } else if (!hu)
        err(0);
      (s4 ? dhu4 : dhu)(dat.subarray(bt, bt += lcs), buf.subarray(spl), hu);
    }
    var ns = dat[bt++];
    if (ns) {
      if (ns == 255)
        ns = (dat[bt++] | dat[bt++] << 8) + 32512;
      else if (ns > 127)
        ns = ns - 128 << 8 | dat[bt++];
      var scm = dat[bt++];
      if (scm & 3)
        err(0);
      var dts = [dmlt, doct, dllt];
      for (var i = 2; i > -1; --i) {
        var md = scm >> (i << 1) + 2 & 3;
        if (md == 1) {
          var rbuf = new u8([0, 0, dat[bt++]]);
          dts[i] = {
            s: rbuf.subarray(2, 3),
            n: rbuf.subarray(0, 1),
            t: new u16(rbuf.buffer, 0, 1),
            b: 0
          };
        } else if (md == 2) {
          _a = rfse(dat, bt, 9 - (i & 1)), bt = _a[0], dts[i] = _a[1];
        } else if (md == 3) {
          if (!st.t)
            err(0);
          dts[i] = st.t[i];
        }
      }
      var _b = st.t = dts, mlt = _b[0], oct = _b[1], llt = _b[2];
      var lb = dat[ebt - 1];
      if (!lb)
        err(0);
      var spos = (ebt << 3) - 8 + msb(lb) - llt.b, cbt = spos >> 3, oubt = 0;
      var lst = (dat[cbt] | dat[cbt + 1] << 8) >> (spos & 7) & (1 << llt.b) - 1;
      cbt = (spos -= oct.b) >> 3;
      var ost = (dat[cbt] | dat[cbt + 1] << 8) >> (spos & 7) & (1 << oct.b) - 1;
      cbt = (spos -= mlt.b) >> 3;
      var mst = (dat[cbt] | dat[cbt + 1] << 8) >> (spos & 7) & (1 << mlt.b) - 1;
      for (++ns; --ns; ) {
        var llc = llt.s[lst];
        var lbtr = llt.n[lst];
        var mlc = mlt.s[mst];
        var mbtr = mlt.n[mst];
        var ofc = oct.s[ost];
        var obtr = oct.n[ost];
        cbt = (spos -= ofc) >> 3;
        var ofp = 1 << ofc;
        var off = ofp + ((dat[cbt] | dat[cbt + 1] << 8 | dat[cbt + 2] << 16 | dat[cbt + 3] << 24) >>> (spos & 7) & ofp - 1);
        cbt = (spos -= mlb[mlc]) >> 3;
        var ml = mlbl[mlc] + ((dat[cbt] | dat[cbt + 1] << 8 | dat[cbt + 2] << 16) >> (spos & 7) & (1 << mlb[mlc]) - 1);
        cbt = (spos -= llb[llc]) >> 3;
        var ll = llbl[llc] + ((dat[cbt] | dat[cbt + 1] << 8 | dat[cbt + 2] << 16) >> (spos & 7) & (1 << llb[llc]) - 1);
        cbt = (spos -= lbtr) >> 3;
        lst = llt.t[lst] + ((dat[cbt] | dat[cbt + 1] << 8) >> (spos & 7) & (1 << lbtr) - 1);
        cbt = (spos -= mbtr) >> 3;
        mst = mlt.t[mst] + ((dat[cbt] | dat[cbt + 1] << 8) >> (spos & 7) & (1 << mbtr) - 1);
        cbt = (spos -= obtr) >> 3;
        ost = oct.t[ost] + ((dat[cbt] | dat[cbt + 1] << 8) >> (spos & 7) & (1 << obtr) - 1);
        if (off > 3) {
          st.o[2] = st.o[1];
          st.o[1] = st.o[0];
          st.o[0] = off -= 3;
        } else {
          var idx = off - (ll != 0);
          if (idx) {
            off = idx == 3 ? st.o[0] - 1 : st.o[idx];
            if (idx > 1)
              st.o[2] = st.o[1];
            st.o[1] = st.o[0];
            st.o[0] = off;
          } else
            off = st.o[0];
        }
        for (var i = 0; i < ll; ++i) {
          buf[oubt + i] = buf[spl + i];
        }
        oubt += ll, spl += ll;
        var stin = oubt - off;
        if (stin < 0) {
          var len = -stin;
          var bs = st.e + stin;
          if (len > ml)
            len = ml;
          for (var i = 0; i < len; ++i) {
            buf[oubt + i] = st.w[bs + i];
          }
          oubt += len, ml -= len, stin = 0;
        }
        for (var i = 0; i < ml; ++i) {
          buf[oubt + i] = buf[stin + i];
        }
        oubt += ml;
      }
      if (oubt != spl) {
        while (spl < buf.length) {
          buf[oubt++] = buf[spl++];
        }
      } else
        oubt = buf.length;
      if (out)
        st.y += oubt;
      else
        buf = slc(buf, 0, oubt);
    } else if (out) {
      st.y += lss;
      if (spl) {
        for (var i = 0; i < lss; ++i) {
          buf[i] = buf[spl + i];
        }
      }
    } else if (spl)
      buf = slc(buf, spl);
    st.b = ebt;
    return buf;
  }
  err(2);
};
var cct = function(bufs, ol) {
  if (bufs.length == 1)
    return bufs[0];
  var buf = new u8(ol);
  for (var i = 0, b = 0; i < bufs.length; ++i) {
    var chk = bufs[i];
    buf.set(chk, b);
    b += chk.length;
  }
  return buf;
};
function decompress(dat, buf) {
  var bufs = [], nb = +!buf;
  var bt = 0, ol = 0;
  for (; dat.length; ) {
    var st = rzfh(dat, nb || buf);
    if (typeof st == "object") {
      if (nb) {
        buf = null;
        if (st.w.length == st.u) {
          bufs.push(buf = st.w);
          ol += st.u;
        }
      } else {
        bufs.push(buf);
        st.e = 0;
      }
      for (; !st.l; ) {
        var blk = rzb(dat, st, buf);
        if (!blk)
          err(5);
        if (buf)
          st.e = st.y;
        else {
          bufs.push(blk);
          ol += blk.length;
          cpw(st.w, 0, blk.length);
          st.w.set(blk, st.w.length - blk.length);
        }
      }
      bt = st.b + st.c * 4;
    } else
      bt = st;
    dat = dat.subarray(bt);
  }
  return cct(bufs, ol);
}

// protocol.ts
var PARAMS_BACKPRESSURE_BYTES = 8 * 1024;
var _fBuf = new ArrayBuffer(4);
var _fU32 = new Uint32Array(_fBuf);
var _fF32 = new Float32Array(_fBuf);
function _half2single(h) {
  const s = (h & 32768) << 16;
  let e = (h & 31744) >> 10;
  let f = h & 1023;
  if (e === 0) {
    if (f === 0) {
      _fU32[0] = s;
      return _fF32[0];
    }
    while ((f & 1024) === 0) {
      f <<= 1;
      e--;
    }
    e++;
    f &= ~1024;
  } else if (e === 31) {
    _fU32[0] = s | 2139095040 | f << 13;
    return _fF32[0];
  }
  e = e + (127 - 15);
  _fU32[0] = s | e << 23 | f << 13;
  return _fF32[0];
}
function float16ArrayToFloat32(u162) {
  const out = new Float32Array(u162.length);
  for (let i = 0; i < u162.length; i++) out[i] = _half2single(u162[i]);
  return out;
}
function packPcmFrame(interleaved, channels) {
  const samples = interleaved.length / channels;
  const hdr = new ArrayBuffer(8);
  const dv = new DataView(hdr);
  dv.setUint32(0, channels, true);
  dv.setUint32(4, samples, true);
  const pcm = new Uint8Array(interleaved.buffer);
  const combined = new Uint8Array(hdr.byteLength + pcm.byteLength);
  combined.set(new Uint8Array(hdr), 0);
  combined.set(pcm, hdr.byteLength);
  return combined;
}
function makeAttemptId() {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch {
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
function hostFromWsUrl(url) {
  try {
    return new URL(url).host;
  } catch {
    return "";
  }
}
var RemoteBackend = class extends EventTarget {
  url;
  ws = null;
  ready = false;
  /** True iff `close()` was called from the app (user-initiated session
   *  teardown). Distinguishes a deliberate disconnect from a network drop
   *  / server crash so the close-event listener can decide whether to
   *  trigger automatic reconnect. */
  closedByUser = false;
  initialBuffer = null;
  duration = 0;
  channels = 0;
  sampleRate = SAMPLE_RATE;
  loraCatalog = [];
  loraDir = "";
  detectedBpm = null;
  detectedKey = null;
  detectedTimeSignature = null;
  /** Active checkpoint identifier (e.g. "acestep-v15-turbo"). Null when
   *  the server didn't ship one (older backend, --no-backend mode). */
  checkpoint = null;
  /** Model-scale label for the active checkpoint ("2B" | "5B" | null).
   *  Used by the LoRA library UI to hide LoRAs whose trained
   *  ``base_model_scale`` doesn't match. Null = unknown checkpoint;
   *  the UI treats that as "don't filter". */
  checkpointScale = null;
  /** Current StreamPipeline ring-buffer depth, mirrored from the
   *  server. Set from the ``ready`` message and from ``depth_applied``
   *  acks after a successful runtime retune. */
  pipelineDepth = null;
  /** Largest depth the server's loaded backend can serve. TRT decoders
   *  report their hidden_states batch_max; eager / compile pin to 4.
   *  Null until ready. */
  maxPipelineDepth = null;
  /** Backend-declared audio geometry from `ready.geometry`. Null on
   *  servers (and recorded replays) from before the backend-seam
   *  contract surface — fall back to the legacy flat ready fields
   *  (duration/channels/sampleRate above) and client constants. */
  geometry = null;
  /** Backend capability mask from `ready.capabilities`. Null = older
   *  server / replay: treat as ungated (everything available). */
  capabilities = null;
  /** Per-session knob manifest from `ready.knob_manifest` — the same
   *  `{version, knobs}` envelope `GET /api/knobs` serves, but resolved
   *  for THIS session (SDE mode, enabled `lora_str_<id>` knobs). Null
   *  on older servers / replays; `/api/knobs` remains the static
   *  pre-session probe. */
  knobManifest = null;
  /** Active manual steering slot count, mirrored from the server
   *  (`ready` + `manual_slot_count` echoes). Null until ready / on
   *  servers without the steering surface. */
  manualSlotCount = null;
  /** Server-imposed cap on manual steering slots. Null until ready. */
  manualSlotCap = null;
  /** Whether the session's checkpoint has steering vectors. The host
   *  hides the steering tiles when false. Null until ready. */
  steeringAvailable = null;
  /** Browser-observed WS lifecycle for this concrete connection attempt. */
  wsTrace;
  /** Pod-side session id from the optional init_ack telemetry message. */
  backendSessionId = null;
  /** Client id echoed in init_ack; mirrors the config client_id. */
  backendClientId = null;
  _pending;
  _pendingSwap = null;
  _pendingStemAssets = null;
  _pendingStemBuffers = {};
  // Slice decoder runs in a worker so fzstd.decompress + float16→float32
  // never block the render loop or input handling. Worker is single-threaded
  // and postMessage is FIFO, so audio slices stay in order.
  _decoderWorker = null;
  _nextDecodeId = 1;
  // Source-buffer epoch. Bumped right before the swap_ready event is
  // dispatched, so any binary slice that arrives at the WS afterwards is
  // tagged for the new buffer. Slices in flight from before the bump
  // (queued in the WS handler ahead of the swap, or sitting in the
  // decoder worker mid-decode) keep their old epoch and get dropped by
  // the listener — without this they'd land in the new track and bleed
  // chunks of the previous song through.
  _sliceEpoch = 0;
  // Cumulative bytes of binary SLICE frames received on this connection
  // (swap buffers and stem payloads excluded — the server counts the
  // same set on its side). Reported as `slice_bytes_rx` with every
  // params message; the server uses sent-minus-acked as its in-flight
  // window and stops emitting slices when the link can't drain them.
  // Without this, a bandwidth-limited path (SSH tunnel, weak uplink)
  // buffers many seconds of slices in socket/tunnel queues the server
  // can't observe, and every slice lands behind the playhead.
  _sliceBytesRx = 0;
  _promptTransform;
  _sliceWorkerUrl;
  constructor(url, interleaved, channels, config, opts = {}) {
    super();
    this.url = url;
    this._pending = { interleaved, channels, config };
    this.wsTrace = {
      attemptId: makeAttemptId(),
      urlHost: hostFromWsUrl(url),
      connectStartAt: null,
      openAt: null,
      configSentAt: null,
      initAckAt: null,
      readyAt: null,
      closeAt: null,
      errorAt: null,
      phase: "idle",
      ready: false,
      closedByUser: false,
      wsReadyState: null,
      closeCode: null,
      closeReason: ""
    };
    this._promptTransform = opts.promptTransform ?? ((tags) => tags);
    this._sliceWorkerUrl = opts.sliceWorkerUrl;
    this._initDecoderWorker();
  }
  _snapshotTrace() {
    return { ...this.wsTrace };
  }
  _updateTrace(patch) {
    this.wsTrace = {
      ...this.wsTrace,
      ...patch,
      ready: this.ready,
      closedByUser: this.closedByUser,
      wsReadyState: this.ws?.readyState ?? null
    };
    const snapshot = this._snapshotTrace();
    this.dispatchEvent(new CustomEvent("ws_trace_update", { detail: snapshot }));
    return snapshot;
  }
  getWsTrace() {
    return this._snapshotTrace();
  }
  _initDecoderWorker() {
    if (typeof Worker === "undefined") return;
    try {
      const worker = this._sliceWorkerUrl !== void 0 ? new Worker(this._sliceWorkerUrl, { type: "module" }) : new Worker(
        new URL("./workers/sliceDecoder.worker.ts", import.meta.url),
        { type: "module" }
      );
      worker.onmessage = (ev) => {
        const msg = ev.data;
        if (!msg || typeof msg !== "object") return;
        if (msg.ok === false) {
          console.error("[protocol] slice decode failed:", msg.error);
          return;
        }
        if (msg.ok !== true) return;
        const slice = {
          flags: msg.flags,
          startSample: msg.startSample,
          numSamples: msg.numSamples,
          channels: msg.channels,
          tickMs: msg.tickMs,
          decMs: msg.decMs,
          numGens: msg.numGens,
          audio: msg.audio,
          epoch: msg.epoch
        };
        this.dispatchEvent(new CustomEvent("slice", { detail: slice }));
      };
      worker.onerror = (e) => {
        console.error("[protocol] slice decoder worker error:", e);
      };
      this._decoderWorker = worker;
    } catch (e) {
      console.warn("[protocol] worker init failed, falling back to main-thread decode:", e);
      this._decoderWorker = null;
    }
  }
  async connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      this._updateTrace({
        connectStartAt: Date.now(),
        openAt: null,
        configSentAt: null,
        initAckAt: null,
        readyAt: null,
        closeAt: null,
        errorAt: null,
        phase: "connecting",
        closeCode: null,
        closeReason: ""
      });
      let phase = "config";
      ws.onopen = () => {
        if (!this._pending) return;
        this._updateTrace({ openAt: Date.now(), phase: "open" });
        ws.send(JSON.stringify(this._pending.config));
        const useServerFixture = this._pending.config.use_server_fixture === true;
        if (!useServerFixture) {
          const { interleaved, channels } = this._pending;
          ws.send(packPcmFrame(interleaved, channels));
        }
        this._updateTrace({ configSentAt: Date.now(), phase: "config_sent" });
        phase = "ready";
      };
      ws.onmessage = (ev) => {
        if (phase === "ready") {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === "init_ack") {
              this.backendSessionId = typeof msg.session_id === "string" ? msg.session_id : null;
              this.backendClientId = typeof msg.client_id === "string" ? msg.client_id : null;
              this._updateTrace({ initAckAt: Date.now(), phase: "init_ack" });
              this.dispatchEvent(new CustomEvent("ws_init_ack", { detail: msg }));
              return;
            }
            if (msg.type === "error") {
              this._updateTrace({ errorAt: Date.now(), phase: "error" });
              reject(
                new Error(
                  msg.message || `Server error: ${msg.code || "unknown"}`
                )
              );
              return;
            }
            if (msg.type !== "ready") {
              this._updateTrace({ errorAt: Date.now(), phase: "error" });
              reject(new Error(`Unexpected init message: ${ev.data}`));
              return;
            }
            this.duration = msg.duration;
            this.channels = msg.channels;
            this.sampleRate = msg.sample_rate;
            this.loraCatalog = msg.lora_catalog || [];
            this.loraDir = msg.lora_dir || "";
            this.detectedBpm = msg.bpm ?? null;
            this.detectedKey = msg.key ?? null;
            this.detectedTimeSignature = msg.time_signature ?? null;
            this.checkpoint = msg.checkpoint ?? null;
            this.checkpointScale = msg.checkpoint_scale ?? null;
            this.pipelineDepth = typeof msg.pipeline_depth === "number" ? msg.pipeline_depth : null;
            this.maxPipelineDepth = typeof msg.max_pipeline_depth === "number" ? msg.max_pipeline_depth : null;
            this.geometry = msg.geometry ?? null;
            this.capabilities = msg.capabilities ?? null;
            this.knobManifest = msg.knob_manifest ?? null;
            this.manualSlotCount = typeof msg.manual_slot_count === "number" ? msg.manual_slot_count : null;
            this.manualSlotCap = typeof msg.manual_slot_cap === "number" ? msg.manual_slot_cap : null;
            this.steeringAvailable = typeof msg.steering_available === "boolean" ? msg.steering_available : null;
            phase = "initial-buffer";
          } catch (e) {
            this._updateTrace({ errorAt: Date.now(), phase: "error" });
            reject(e);
          }
          return;
        }
        if (phase === "initial-buffer") {
          const u162 = new Uint16Array(ev.data);
          this.initialBuffer = float16ArrayToFloat32(u162);
          this.ready = true;
          this._updateTrace({ readyAt: Date.now(), phase: "ready" });
          phase = "streaming";
          this._pending = null;
          resolve(this);
          this.dispatchEvent(new CustomEvent("ready"));
          this._updateTrace({ phase: "streaming" });
          return;
        }
        if (this._pendingSwap && ev.data instanceof ArrayBuffer) {
          const u162 = new Uint16Array(ev.data);
          const interleaved = float16ArrayToFloat32(u162);
          const meta = this._pendingSwap;
          this._pendingSwap = null;
          this.duration = meta.duration;
          this.channels = meta.channels;
          this._sliceEpoch++;
          this.dispatchEvent(
            new CustomEvent("swap_ready", {
              detail: { ...meta, interleaved }
            })
          );
          return;
        }
        if (this._pendingStemAssets && ev.data instanceof ArrayBuffer) {
          const meta = this._pendingStemAssets;
          const stem = meta.stems[Object.keys(this._pendingStemBuffers).length];
          if (stem) {
            const u162 = new Uint16Array(ev.data);
            this._pendingStemBuffers[stem] = float16ArrayToFloat32(u162);
          }
          const complete = meta.stems.every(
            (name) => this._pendingStemBuffers[name]
          );
          if (complete) {
            const buffers = this._pendingStemBuffers;
            this._pendingStemAssets = null;
            this._pendingStemBuffers = {};
            this.dispatchEvent(
              new CustomEvent("stem_assets", {
                detail: { ...meta, buffers }
              })
            );
          }
          return;
        }
        if (typeof ev.data === "string") {
          let msg;
          try {
            msg = JSON.parse(ev.data);
          } catch {
            return;
          }
          switch (msg.type) {
            case "params_update":
              this.dispatchEvent(
                new CustomEvent("params", { detail: msg.params })
              );
              break;
            case "params_echo":
              this.dispatchEvent(
                new CustomEvent("params_echo", { detail: msg.raw })
              );
              break;
            case "prompt_blend_echo":
              this.dispatchEvent(
                new CustomEvent("prompt_blend_echo", { detail: msg.value })
              );
              break;
            case "prompt_applied":
              this.dispatchEvent(
                new CustomEvent("prompt_applied", { detail: msg.tags })
              );
              break;
            case "lora_catalog":
              this.loraCatalog = msg.catalog || [];
              this.dispatchEvent(
                new CustomEvent("lora_catalog", { detail: this.loraCatalog })
              );
              break;
            case "swap_ready":
              this._pendingSwap = msg;
              break;
            case "swap_failed":
              this.dispatchEvent(
                new CustomEvent("swap_failed", { detail: msg.error })
              );
              break;
            case "stem_assets":
              this._pendingStemAssets = msg;
              this._pendingStemBuffers = {};
              break;
            case "stem_failed":
              this._pendingStemAssets = null;
              this._pendingStemBuffers = {};
              this.dispatchEvent(
                new CustomEvent("stem_failed", { detail: msg })
              );
              break;
            case "timbre_set":
              this.dispatchEvent(
                new CustomEvent("timbre_set", { detail: msg })
              );
              break;
            case "timbre_cleared":
              this.dispatchEvent(new CustomEvent("timbre_cleared"));
              break;
            case "timbre_failed":
              this.dispatchEvent(
                new CustomEvent("timbre_failed", { detail: msg.error })
              );
              break;
            case "structure_set":
              this.dispatchEvent(
                new CustomEvent("structure_set", { detail: msg })
              );
              break;
            case "structure_cleared":
              this.dispatchEvent(new CustomEvent("structure_cleared"));
              break;
            case "structure_failed":
              this.dispatchEvent(
                new CustomEvent("structure_failed", { detail: msg.error })
              );
              break;
            case "depth_applied": {
              const v = typeof msg.value === "number" ? msg.value : null;
              if (v !== null) {
                this.pipelineDepth = v;
                this.dispatchEvent(
                  new CustomEvent("depth_applied", { detail: v })
                );
              }
              break;
            }
            case "command_failed":
              console.warn(
                `[protocol] command_failed: ${msg.command} needs backend capability '${msg.requires}'` + (msg.error ? ` \u2014 ${msg.error}` : "")
              );
              this.dispatchEvent(
                new CustomEvent("command_failed", { detail: msg })
              );
              break;
            case "manual_slot_count": {
              const v = typeof msg.count === "number" ? msg.count : null;
              this.manualSlotCount = v;
              this.dispatchEvent(
                new CustomEvent("manual_slot_count", { detail: v })
              );
              break;
            }
            case "error":
              console.error(
                `[protocol] server error: ${msg.code || "unknown"}` + (msg.message ? ` \u2014 ${msg.message}` : "")
              );
              this.dispatchEvent(
                new CustomEvent("server_error", { detail: msg })
              );
              break;
            default:
              this.dispatchEvent(new CustomEvent("json", { detail: msg }));
          }
          return;
        }
        if (this._decoderWorker) {
          const buf = ev.data;
          this._sliceBytesRx += buf.byteLength;
          this._decoderWorker.postMessage(
            {
              id: this._nextDecodeId++,
              buffer: buf,
              epoch: this._sliceEpoch
            },
            [buf]
          );
        } else {
          try {
            this._sliceBytesRx += ev.data.byteLength;
            const slice = this._parseSlice(ev.data);
            if (slice) {
              slice.epoch = this._sliceEpoch;
              this.dispatchEvent(new CustomEvent("slice", { detail: slice }));
            }
          } catch (e) {
            console.error("[protocol] slice parse failed:", e);
          }
        }
      };
      ws.onerror = (e) => {
        console.error("[protocol] ws error", e);
        const trace = this._updateTrace({
          errorAt: Date.now(),
          phase: this.ready ? this.wsTrace.phase : "error"
        });
        if (!this.ready) {
          reject(
            new Error(
              "WebSocket connection failed (network / port unreachable)"
            )
          );
        }
        this.dispatchEvent(new CustomEvent("ws_connect_error", { detail: trace }));
        this.dispatchEvent(new CustomEvent("error", { detail: e }));
      };
      ws.onclose = (e) => {
        if (!this.ready) {
          let msg;
          if (e.code === PREEMPTED_CLOSE_CODE) {
            msg = "Another connection took over this session.";
          } else if (e.code === 1011) {
            msg = "Session failed while starting \u2014 refresh the page to retry.";
          } else if (e.code === 1006) {
            msg = "Connection lost \u2014 refresh to retry.";
          } else {
            const reason = e.reason || `code ${e.code}`;
            msg = `Connection failed (${reason}) \u2014 refresh to retry.`;
          }
          reject(new Error(msg));
        }
        const trace = this._updateTrace({
          closeAt: Date.now(),
          phase: "closed",
          closeCode: e.code,
          closeReason: e.reason || ""
        });
        this.dispatchEvent(new CustomEvent("ws_close", { detail: trace }));
        this.dispatchEvent(new CustomEvent("close", { detail: e }));
      };
    });
  }
  _parseSlice(buf) {
    if (buf.byteLength < SLICE_HDR_SIZE) return null;
    const dv = new DataView(buf);
    let o = 0;
    const flags = dv.getUint8(o);
    o += 1;
    const startSample = dv.getUint32(o, true);
    o += 4;
    const numSamples = dv.getUint32(o, true);
    o += 4;
    const channels = dv.getUint16(o, true);
    o += 2;
    const tickMs = dv.getFloat32(o, true);
    o += 4;
    const decMs = dv.getFloat32(o, true);
    o += 4;
    const numGens = dv.getUint32(o, true);
    o += 4;
    let payload = new Uint8Array(buf, SLICE_HDR_SIZE);
    if (flags === SLICE_FLAG_DELTA) {
      payload = decompress(payload);
    }
    const aligned = new ArrayBuffer(payload.byteLength);
    new Uint8Array(aligned).set(payload);
    const u162 = new Uint16Array(aligned);
    const audio = float16ArrayToFloat32(u162);
    return {
      flags,
      startSample,
      numSamples,
      channels,
      tickMs,
      decMs,
      numGens,
      audio,
      // Caller (the WS onmessage fallback path) overwrites this with the
      // current source epoch right before dispatching.
      epoch: 0
    };
  }
  /** Returns true only when the message was actually handed to `ws.send`.
   *  Callers that consume a one-shot sample (e.g. the worst-slice-lead
   *  tracker, which clears on read) must re-arm it when this returns false,
   *  or the sample is lost on a dropped tick. */
  sendParams(raw, playbackPos, sliceLeadS) {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    if (this.ws.bufferedAmount > PARAMS_BACKPRESSURE_BYTES) return false;
    try {
      const msg = {
        type: "params",
        raw,
        playback_pos: playbackPos,
        // Monotonic send stamp; the server pairs it with arrival time to
        // estimate report staleness for queueing the gate above can't see
        // (middlebox/tunnel buffering, server-side recv backlog).
        client_time: performance.now() / 1e3,
        // Flow-control ack: cumulative slice bytes received. The server
        // holds back slice emission while sent-minus-acked exceeds its
        // in-flight window, so a slow link gets fresh slices at link
        // rate instead of an ever-staler backlog.
        slice_bytes_rx: this._sliceBytesRx
      };
      if (sliceLeadS !== void 0 && Number.isFinite(sliceLeadS)) {
        msg.slice_lead_s = sliceLeadS;
      }
      this.ws.send(JSON.stringify(msg));
      return true;
    } catch {
      return false;
    }
  }
  sendPrompt(tags, key, timeSignature, tagsB) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = {
        type: "prompt",
        tags: this._promptTransform(tags)
      };
      if (tagsB) msg.tags_b = this._promptTransform(tagsB);
      if (key) msg.key = key;
      if (timeSignature) msg.time_signature = timeSignature;
      this.ws.send(JSON.stringify(msg));
      if (typeof window !== "undefined" && window.__demonPromptLog) {
        console.log(
          `[demon prompt \u2192 engine]
  tags A (wire)  : ${JSON.stringify(msg.tags)}
  tags B (wire)  : ${msg.tags_b != null ? JSON.stringify(msg.tags_b) : "(none)"}`
        );
      }
    } catch {
    }
  }
  /**
   * Live prompt A/B blend knob. Backend keeps cached cond pairs for both
   * prompts (encoded by the most recent ``sendPrompt`` that carried a
   * ``tags_b``) and lerps between them by `value` ∈ [0,1] — 0 == A, 1 == B.
   * Same shape as ``sendSetTimbreStrength``; cheap per slider tick.
   */
  sendSetPromptBlend(value) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = {
        type: "set_prompt_blend",
        value: Math.max(0, Math.min(1, value))
      };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /**
   * Switch the interpolation path for one of the four live blends
   * (prompt / timbre / structure / feedback) between "slerp" and
   * "linear". slerp walks the per-frame geodesic so the blended value's
   * norm stays constant across the sweep; linear is a straight average
   * that dips at the midpoint. The server applies it immediately
   * (prompt/timbre recompute the cached conditioning; structure/feedback
   * are read live each tick), so the change is audible without a
   * restart. Discrete setting, so no smoothing/echo channel.
   */
  sendSetInterpMethod(path, method) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = {
        type: "set_interp_method",
        path,
        method
      };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /**
   * Live pipeline_depth retune. The server stages the value and applies
   * it on the next runner-thread before_tick rendezvous, then echoes
   * the (clamped) result back as ``depth_applied``. Shrinking discards
   * in-flight slots beyond the new depth; growing extends with empty
   * slots that warm up over the next ``newDepth - oldDepth`` ticks.
   */
  sendSetDepth(value) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    if (!Number.isFinite(value)) return;
    try {
      const msg = {
        type: "set_depth",
        value: Math.round(value)
      };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /**
   * Mirror the client loop band to the server. The worklet already wraps
   * end→start locally; this tells the pipeline so it wraps its predictive
   * decode target inside the band too, regenerating the seam after `start`
   * before the playhead loops back to it instead of leaving one stale
   * window of pre-change audio at every loop restart. Pass `null`s to
   * clear (linear chase resumes). Seconds, matching `playback_pos`.
   */
  sendLoopBand(startSec, endSec) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = {
        type: "loop_band",
        start_sec: startSec,
        end_sec: endSec
      };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  sendEnableLora(id, strength) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = {
        type: "enable_lora",
        id
      };
      if (typeof strength === "number") msg.strength = strength;
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  sendDisableLora(id) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = { type: "disable_lora", id };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /** Add the next manual steering slot (LIFO). Server echoes
   *  ``manual_slot_count`` on success or refusal. */
  sendManualSlotAdd() {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = { type: "manual_slot_add" };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /** Pop the highest-numbered manual steering slot. */
  sendManualSlotPop() {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = { type: "manual_slot_pop" };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /**
   * Live timbre-strength knob. Backend keeps a cached
   * (cond_silence, cond_full) pair and lerp-blends their encoder hidden
   * states by `value` ∈ [0,1] — 1.0 == full timbre reference, 0.0 ==
   * silence-baseline timbre. Cheap enough to send per slider tick.
   */
  sendSetTimbreStrength(value) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = {
        type: "set_timbre_strength",
        value: Math.max(0, Math.min(1, value))
      };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /**
   * Send a typed JSON header followed by a binary audio frame
   * (packPcmFrame). Used by the timbre/structure source uploads and
   * swap_source; the caller builds the typed command so the header is
   * contract-checked at compile time.
   */
  sendAudioFrame(msg, interleaved, channels) {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      this.ws.send(JSON.stringify(msg));
      this.ws.send(packPcmFrame(interleaved, channels));
      return true;
    } catch (e) {
      console.error(`[protocol] ${msg.type} failed:`, e);
      return false;
    }
  }
  /**
   * Upload an audio clip as the active timbre reference. Server VAE-
   * encodes it and replaces cond_full with one conditioned on the clip's
   * latent. The clip is capped server-side to the playback source's
   * duration to fit the loaded TRT profile. Replies with timbre_set on
   * success or timbre_failed on error.
   */
  sendSetTimbreSource(interleaved, channels, name) {
    return this.sendAudioFrame(
      { type: "set_timbre_source", name },
      interleaved,
      channels
    );
  }
  /**
   * Pick a Library fixture as the active timbre reference. The server
   * resolves the WAV from its local HF cache and runs the same apply
   * path as a PCM upload, so the browser doesn't have to fetch +
   * decode + re-upload a file that already lives on the pod's disk.
   * Replies with timbre_set on success or timbre_failed on error
   * (e.g. unknown fixture name).
   */
  sendSetTimbreFixture(name) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = { type: "set_timbre_fixture", name };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /**
   * Drop the active timbre reference; server falls back to self-timbre
   * (encode against the playback source's own latent). Replies with
   * timbre_cleared on success.
   */
  sendClearTimbreSource() {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = { type: "clear_timbre_source" };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /**
   * Upload an audio clip as the active structure (semantic-hint)
   * reference. Server pads/trims it to match the playback source's
   * exact sample count, runs prepare_source to extract the override's
   * context_latent, and replaces stream.source.context_latent so the
   * runner's hint-strength blend reads the new structure. Replies with
   * structure_set on success or structure_failed on error.
   */
  sendSetStructureSource(interleaved, channels, name) {
    return this.sendAudioFrame(
      { type: "set_structure_source", name },
      interleaved,
      channels
    );
  }
  /**
   * Pick a Library fixture as the active structure reference. Server-
   * side counterpart to sendSetTimbreFixture: avoids the wasteful
   * fetch+decode+upload round trip for fixtures that already live on
   * the pod's disk. Replies with structure_set / structure_failed.
   */
  sendSetStructureFixture(name) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = {
        type: "set_structure_fixture",
        name
      };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /**
   * Drop the active structure reference; server restores the playback
   * source's own context_latent. Replies with structure_cleared.
   */
  sendClearStructureSource() {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    try {
      const msg = {
        type: "clear_structure_source"
      };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /**
   * Replace the source audio in-flight. Server pauses generation, re-runs
   * prepare_source / encode_text on the new waveform, then replies with
   * swap_ready + a binary buffer (handled in onmessage).
   */
  sendSwapSource(interleaved, channels, tags, key, fixtureName, timeSignature, stemSourceMode) {
    const msg = {
      type: "swap_source"
    };
    if (tags) msg.tags = tags;
    if (key) msg.key = key;
    if (fixtureName) msg.fixture_name = fixtureName;
    if (timeSignature) msg.time_signature = timeSignature;
    if (stemSourceMode) msg.stem_source_mode = stemSourceMode;
    return this.sendAudioFrame(msg, interleaved, channels);
  }
  /**
   * Swap to a source that already lives on the pod (a built-in fixture or
   * a persisted upload), identified by name only — NO PCM is sent. The
   * server loads the waveform off its own disk, which lets the sidecar +
   * stem caches hit instead of re-encoding and re-ripping a re-uploaded
   * buffer. The reply is the same swap_ready + binary buffer as
   * sendSwapSource, so the player gets its crossfade buffer from the
   * server echo.
   */
  sendSwapSourceByName(fixtureName, tags, key, timeSignature, stemSourceMode) {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    try {
      const msg = {
        type: "swap_source",
        use_server_source: true,
        fixture_name: fixtureName
      };
      if (tags) msg.tags = tags;
      if (key) msg.key = key;
      if (timeSignature) msg.time_signature = timeSignature;
      if (stemSourceMode) msg.stem_source_mode = stemSourceMode;
      this.ws.send(JSON.stringify(msg));
      return true;
    } catch (e) {
      console.error("[protocol] sendSwapSourceByName failed:", e);
      return false;
    }
  }
  close() {
    this.closedByUser = true;
    this._updateTrace({ closedByUser: true });
    try {
      this.ws?.close();
    } catch {
    }
    try {
      this._decoderWorker?.terminate();
    } catch {
    }
    this._decoderWorker = null;
  }
  /** Align the slice-epoch counter to a target value. Used by the
   *  reconnect path: after `player.swap()` bumps `player.swapCount`
   *  to mark a fresh source buffer, the new remote's `_sliceEpoch`
   *  (which starts at 0 for every new `RemoteBackend` instance) has
   *  to match — otherwise the slice listener's `epoch !== swapCount`
   *  guard drops every incoming slice for the rest of the session.
   *  Safe to call before any WS slice has been posted to the
   *  decoder worker (which is the case during reconnect, since
   *  worker post happens inside `ws.onmessage` after `connect()`
   *  resolves and the slice listener can run). */
  setSliceEpoch(epoch) {
    this._sliceEpoch = epoch;
  }
  /** Test/dev hook: synthesize an abnormal close so the client-side
   *  reconnect path can be exercised without needing real network
   *  failure. The browser maps a TCP RST (the dominant production
   *  cause of 1006 from RunPod / vast.ai tunnels) to a CloseEvent
   *  with code 1006, wasClean:false. We construct the same event
   *  shape and route it through the same `close` listeners the real
   *  socket would, then tear down the underlying ws so no further
   *  frames or events arrive — matching what the OS-level RST does.
   */
  simulateClose(code = 1006, reason = "simulated") {
    const ws = this.ws;
    if (ws) {
      try {
        ws.onclose = null;
        ws.onerror = null;
        ws.onmessage = null;
      } catch {
      }
      try {
        ws.close();
      } catch {
      }
    }
    this.ws = null;
    const trace = this._updateTrace({
      closeAt: Date.now(),
      phase: "closed",
      closeCode: code,
      closeReason: reason
    });
    this.dispatchEvent(new CustomEvent("ws_close", { detail: trace }));
    let ev;
    try {
      ev = new CloseEvent("close", { code, reason, wasClean: false });
    } catch {
      const e = new Event("close");
      e.code = code;
      e.reason = reason;
      e.wasClean = false;
      ev = e;
    }
    this.dispatchEvent(new CustomEvent("close", { detail: ev }));
  }
};

// audio/lufs.ts
var PRE_B = [1.53512485958697, -2.69169618940638, 1.19839281085285];
var PRE_A = [-1.69065929318241, 0.73248077421585];
var RLB_B = [1, -2, 1];
var RLB_A = [-1.99004745483398, 0.99007225036621];
var A_SECTIONS = [
  [0.2343017922995135, 0.4686035845990268, 0.2343017922995133, -0.2245584580597783, 0.0126066252715464],
  [1, -1.9999999663681969, 0.9999999830617619, -1.8938704944148377, 0.8951597688151856],
  [1, -2.00000003363181, 1.0000000169382441, -1.9946144563012589, 0.9946217073246171]
];
var BLOCK_S = 0.4;
var HOP_S = 0.1;
var ABSOLUTE_GATE_LKFS = -70;
var RELATIVE_GATE_OFFSET = 10;
function biquadInPlace(x, bn, an) {
  const [b0, b1, b2] = bn;
  const [a1, a2] = an;
  let z1 = 0;
  let z2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    const y = b0 * v + z1;
    z1 = b1 * v - a1 * y + z2;
    z2 = b2 * v - a2 * y;
    x[i] = y;
  }
}
function kWeight(x) {
  biquadInPlace(x, PRE_B, PRE_A);
  biquadInPlace(x, RLB_B, RLB_A);
}
function aWeight(x) {
  for (const sec of A_SECTIONS) {
    const [b0, b1, b2, a1, a2] = sec;
    biquadInPlace(x, [b0, b1, b2], [a1, a2]);
  }
}
function deinterleave(interleaved, channels, frames) {
  const out = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Float32Array(frames);
    for (let i = 0; i < frames; i++) ch[i] = interleaved[i * channels + c];
    out.push(ch);
  }
  return out;
}
function samplePeak(interleaved) {
  let peak = 0;
  for (let i = 0; i < interleaved.length; i++) {
    const a = Math.abs(interleaved[i]);
    if (a > peak) peak = a;
  }
  return peak;
}
function blockMeanSquares(perChannel, frames, blockSize, hopSize) {
  const numBlocks = Math.max(0, Math.floor((frames - blockSize) / hopSize) + 1);
  const out = new Float64Array(numBlocks);
  for (let b = 0; b < numBlocks; b++) {
    const start = b * hopSize;
    let weighted = 0;
    for (let c = 0; c < perChannel.length; c++) {
      const ch = perChannel[c];
      let ms = 0;
      for (let i = start; i < start + blockSize; i++) ms += ch[i] * ch[i];
      weighted += ms / blockSize;
    }
    out[b] = weighted;
  }
  return out;
}
function gatedMean(blocks, startIdx, endIdx, toLkfs) {
  if (endIdx <= startIdx) return null;
  let absSum = 0;
  let absCount = 0;
  for (let i = startIdx; i < endIdx; i++) {
    const ms = blocks[i];
    if (ms > 0 && toLkfs(ms) >= ABSOLUTE_GATE_LKFS) {
      absSum += ms;
      absCount++;
    }
  }
  if (absCount === 0) return null;
  const meanAbs = absSum / absCount;
  const relThreshold = toLkfs(meanAbs) - RELATIVE_GATE_OFFSET;
  let relSum = 0;
  let relCount = 0;
  for (let i = startIdx; i < endIdx; i++) {
    const ms = blocks[i];
    if (ms > 0 && toLkfs(ms) >= ABSOLUTE_GATE_LKFS && toLkfs(ms) >= relThreshold) {
      relSum += ms;
      relCount++;
    }
  }
  if (relCount === 0) return null;
  return relSum / relCount;
}
function lkfsLufs(meanSquare) {
  return -0.691 + 10 * Math.log10(meanSquare);
}
function lkfsDba(meanSquare) {
  return 10 * Math.log10(meanSquare);
}
function measureIntegratedLufs(interleaved, channels, sampleRate) {
  if (sampleRate !== 48e3) return { integratedLufs: null, peak: 0 };
  const peak = samplePeak(interleaved);
  const frames = interleaved.length / channels | 0;
  const blockSize = Math.round(BLOCK_S * sampleRate);
  const hopSize = Math.round(HOP_S * sampleRate);
  if (frames < blockSize) return { integratedLufs: null, peak };
  const perChannel = deinterleave(interleaved, channels, frames);
  for (const ch of perChannel) kWeight(ch);
  const blocks = blockMeanSquares(perChannel, frames, blockSize, hopSize);
  const meanRel = gatedMean(blocks, 0, blocks.length, lkfsLufs);
  if (meanRel === null) return { integratedLufs: null, peak };
  return { integratedLufs: lkfsLufs(meanRel), peak };
}
function measureIntegratedDba(interleaved, channels, sampleRate) {
  if (sampleRate !== 48e3) return { integratedDba: null, peak: 0 };
  const peak = samplePeak(interleaved);
  const frames = interleaved.length / channels | 0;
  const blockSize = Math.round(BLOCK_S * sampleRate);
  const hopSize = Math.round(HOP_S * sampleRate);
  if (frames < blockSize) return { integratedDba: null, peak };
  const perChannel = deinterleave(interleaved, channels, frames);
  for (const ch of perChannel) aWeight(ch);
  const blocks = blockMeanSquares(perChannel, frames, blockSize, hopSize);
  const meanRel = gatedMean(blocks, 0, blocks.length, lkfsDba);
  if (meanRel === null) return { integratedDba: null, peak };
  return { integratedDba: lkfsDba(meanRel), peak };
}
function measureLoudness(interleaved, channels, sampleRate, metric) {
  if (metric === "dba") {
    const m2 = measureIntegratedDba(interleaved, channels, sampleRate);
    return { value: m2.integratedDba, peak: m2.peak };
  }
  const m = measureIntegratedLufs(interleaved, channels, sampleRate);
  return { value: m.integratedLufs, peak: m.peak };
}
function findLoudestShortTermLoudness(interleaved, channels, sampleRate, windowSec, metric) {
  if (sampleRate !== 48e3) return null;
  const frames = interleaved.length / channels | 0;
  const blockSize = Math.round(BLOCK_S * sampleRate);
  const hopSize = Math.round(HOP_S * sampleRate);
  const winFrames = Math.round(windowSec * sampleRate);
  if (frames < winFrames) return null;
  const perChannel = deinterleave(interleaved, channels, frames);
  if (metric === "dba") {
    for (const ch of perChannel) aWeight(ch);
  } else {
    for (const ch of perChannel) kWeight(ch);
  }
  const blocks = blockMeanSquares(perChannel, frames, blockSize, hopSize);
  const blocksPerWindow = Math.max(
    1,
    Math.floor((winFrames - blockSize) / hopSize) + 1
  );
  const winHopBlocks = Math.max(1, Math.round(0.25 / HOP_S));
  const toLkfs = metric === "dba" ? lkfsDba : lkfsLufs;
  let best = null;
  for (let i = 0; i + blocksPerWindow <= blocks.length; i += winHopBlocks) {
    const meanRel = gatedMean(blocks, i, i + blocksPerWindow, toLkfs);
    if (meanRel === null) continue;
    const v = toLkfs(meanRel);
    if (best === null || v > best) best = v;
  }
  return best;
}
function lufsMakeupGain(integratedLufs, targetLufs, peak, peakCeiling) {
  if (peak <= 0) return 1;
  const desired = Math.pow(10, (targetLufs - integratedLufs) / 20);
  const peakClamp = peakCeiling / peak;
  return Math.min(desired, peakClamp);
}
function measureBlock(interleaved, channels, metric) {
  if (interleaved.length === 0) {
    return { loudness: Number.NEGATIVE_INFINITY, peak: 0 };
  }
  const frames = interleaved.length / channels | 0;
  let peak = 0;
  for (let i = 0; i < interleaved.length; i++) {
    const a = Math.abs(interleaved[i]);
    if (a > peak) peak = a;
  }
  const perChannel = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Float32Array(frames);
    for (let i = 0; i < frames; i++) ch[i] = interleaved[i * channels + c];
    perChannel.push(ch);
  }
  if (metric === "dba") {
    for (const ch of perChannel) aWeight(ch);
  } else {
    for (const ch of perChannel) kWeight(ch);
  }
  let weightedMs = 0;
  for (const ch of perChannel) {
    let ms = 0;
    for (let i = 0; i < frames; i++) ms += ch[i] * ch[i];
    weightedMs += frames > 0 ? ms / frames : 0;
  }
  if (weightedMs <= 0) {
    return { loudness: Number.NEGATIVE_INFINITY, peak };
  }
  const offset = metric === "dba" ? 0 : -0.691;
  return { loudness: offset + 10 * Math.log10(weightedMs), peak };
}

// audio/AudioPlayer.ts
var DEFAULT_WORKLET_URL = "/audio-worklet.js?v=5";
var SP_OVERLAY_VOL_RAMP_S = 0.025;
var LUFS_PEAK_CEILING = 0.891;
var LUFS_PEAK_HEADROOM_DEFAULT = 4;
var LUFS_GAIN_RAMP_TC = 0.08;
var LUFS_METER_WINDOW_DEFAULT_SEC = 3;
var LUFS_METER_WINDOW_MIN_SEC = 0.5;
var LUFS_METER_INTERVAL_MS = 100;
var LUFS_CHUNK_FRAMES = 14400;
var LUFS_SILENCE_FLOOR_DB_DEFAULT = 30;
var LUFS_SILENCE_FLOOR_HYSTERESIS_DB_DEFAULT = 6;
var AudioPlayer = class {
  _loudnessConfig;
  _workletUrl;
  constructor(opts = {}) {
    this._loudnessConfig = opts.loudnessConfig ?? (() => ({}));
    this._workletUrl = opts.workletUrl ?? DEFAULT_WORKLET_URL;
  }
  ctx = null;
  node = null;
  positionSec = 0;
  swapCount = 0;
  channels = 2;
  frameCount = 0;
  // Most recent kick (RMS over a 480-frame window, soft-clipped to [0,1]).
  // Computed by the AudioWorklet on the audio thread and posted alongside
  // position; the main render path reads it via getKick(). On the
  // ScriptProcessor fallback path this stays 0 — kick reactivity degrades
  // gracefully (no flashes on beats) rather than blocking the main thread
  // with a per-frame RMS loop. See PERFORMANCE.md.
  kick = 0;
  _listeners = /* @__PURE__ */ new Set();
  // Trailing-throttle handle for mirror-change notifications. Slices
  // stream in at up to ~40-50/s and every patch used to fire the full
  // listener set synchronously (waveform recompute, HUD refresh, curve
  // overlays) — ~tens of ms of main-thread work PER SLICE. When the tab
  // is busy or background-throttled that made slice-apply slower than
  // slice production, so the receive queue grew without bound and
  // patches landed ever further behind the playhead (heard as the raw
  // source bleeding through). Subscribers are visual consumers; 10 Hz
  // is plenty. swap() still notifies synchronously (rare + structural).
  _notifyTimer = null;
  _mirror = null;
  _useWorklet = false;
  _spBuffer = null;
  _spPosition = 0;
  _recordDest = null;
  _masterOut = null;
  _stemOverlays = {
    vocals: { interleaved: null, channels: 2, frameCount: 0, targetVolume: 0, volume: 0 },
    instruments: { interleaved: null, channels: 2, frameCount: 0, targetVolume: 0, volume: 0 }
  };
  _spOverlayVolAlpha = 0;
  // Loop + seek state. The worklet path owns its own copy of `loop` (set
  // via postMessage); these fields are the main-thread mirror so the SP
  // fallback path can read them directly. _spEndSignaled is the SP-path
  // equivalent of the worklet's _endSignaled one-shot.
  _loop = true;
  _spEndSignaled = false;
  _endOfBufferListeners = /* @__PURE__ */ new Set();
  // Loudness matching: a GainNode sits between the worklet and
  // destination. We measure the source's integrated loudness once at
  // init() / swap() and lock it as the target. The meter periodically
  // measures the playhead window; if that window is quieter than the
  // source target, we boost it up. We never attenuate. So source
  // plays at unity (it already is the target) and any quieter
  // remix-output at the playhead gets boosted up to source loudness.
  //
  // No running-max, no high-water bookkeeping: the source is the
  // reference, full stop. If the operator's remix happens to be
  // louder than source, "louder side wins" via the never-attenuate
  // clamp -- it plays at unity, source plays at unity, and the
  // matcher does nothing. That's the intended behaviour.
  _makeupGain = null;
  _lufsEnabled = false;
  _sourceTarget = null;
  _meterIntervalId = null;
  _meterWindowSec = LUFS_METER_WINDOW_DEFAULT_SEC;
  _loudnessMetric = "lufs";
  // Effective peak ceiling for the makeup gain. Default is -1 dBTP
  // (0.891), expanded at init/swap to max(default, source_peak *
  // headroom_factor) so a source with hot peaks doesn't pin the
  // never-attenuate clamp at unity for the entire session.
  _peakCeiling = LUFS_PEAK_CEILING;
  _silenceFloorDb = LUFS_SILENCE_FLOOR_DB_DEFAULT;
  _silenceFloorHysteresisDb = LUFS_SILENCE_FLOOR_HYSTERESIS_DB_DEFAULT;
  _inSilence = false;
  // Per-chunk loudness/peak map of the mirror. The matcher consults
  // these arrays at meter time to know "what's at the playhead right
  // now" without waiting for a sliding window to fill -- which means
  // gain updates the instant the playhead enters a freshly-written
  // denoised chunk, instead of swelling up over the window length.
  // Both arrays are length ceil(totalFrames / LUFS_CHUNK_FRAMES) and
  // are populated at init/swap and refreshed on every patch/addDelta.
  _chunkLoudness = null;
  _chunkPeak = null;
  /** Most recent short-term loudness reading at the playhead (or null
   *  when the window had nothing audible). Units depend on the active
   *  metric (LUFS or dBA). Exposed for UI readouts. */
  lufsMeasured = null;
  get duration() {
    return this.frameCount / SAMPLE_RATE;
  }
  async init(initialBufferInterleaved, channels) {
    this.ctx = new AudioContext({
      sampleRate: SAMPLE_RATE,
      latencyHint: "interactive"
    });
    this.channels = channels;
    this.frameCount = initialBufferInterleaved.length / channels;
    this._mirror = initialBufferInterleaved.slice();
    this._useWorklet = !!this.ctx.audioWorklet;
    if (this._useWorklet) {
      await this.ctx.audioWorklet.addModule(this._workletUrl);
      const node = new AudioWorkletNode(this.ctx, "realtime-buffer", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [channels]
      });
      this.node = node;
      node.port.onmessage = (e) => {
        const msg = e.data;
        if (msg.type === "position") {
          this.positionSec = msg.positionSec ?? 0;
          this.swapCount = msg.swapCount ?? this.swapCount;
          if (typeof msg.kick === "number") this.kick = msg.kick;
        } else if (msg.type === "endOfBuffer") {
          for (const fn of this._endOfBufferListeners) fn();
        }
      };
      const send = initialBufferInterleaved.slice();
      node.port.postMessage(
        { type: "init", buffer: send, channels },
        [send.buffer]
      );
    } else {
      console.warn(
        "[AudioPlayer] AudioWorklet unavailable (non-secure context). Using ScriptProcessor fallback."
      );
      this._spBuffer = initialBufferInterleaved.slice();
      this._spPosition = 0;
      const BUFFER_SIZE = 4096;
      const sp = this.ctx.createScriptProcessor(BUFFER_SIZE, 0, channels);
      this.node = sp;
      sp.onaudioprocess = (e) => this._spProcess(e);
    }
    this._spOverlayVolAlpha = 1 - Math.exp(-1 / (this.ctx.sampleRate * SP_OVERLAY_VOL_RAMP_S));
    this._masterOut = this.ctx.createGain();
    this._masterOut.gain.value = 1;
    this._masterOut.connect(this.ctx.destination);
    this._makeupGain = this.ctx.createGain();
    this._makeupGain.gain.value = 1;
    this.node.connect(this._makeupGain);
    this._makeupGain.connect(this._masterOut);
    this._measureSourceTarget();
    if (this._lufsEnabled) this._startMetering();
  }
  /** Overwrite a region of the worklet's buffer. */
  patch(startFrame, audioInterleaved) {
    this._writeMirror(startFrame, audioInterleaved, false);
    this._refreshChunks(startFrame, audioInterleaved.length / this.channels);
    if (this._useWorklet && this.node) {
      const send = audioInterleaved.slice();
      this.node.port.postMessage(
        { type: "patch", start: startFrame, audio: send },
        [send.buffer]
      );
    } else {
      this._writeSPBuffer(startFrame, audioInterleaved, false);
    }
  }
  /**
   * Replace the entire loop buffer. The worklet crossfades old → new over
   * CROSSFADE_SECONDS (25 ms); ScriptProcessor fallback does an instant
   * swap (the seam-fade still hides the wrap).
   */
  swap(interleavedBuffer, channels) {
    this.clearStemOverlays();
    this.channels = channels || this.channels;
    this.frameCount = interleavedBuffer.length / this.channels;
    this._mirror = interleavedBuffer.slice();
    this.swapCount++;
    for (const fn of this._listeners) fn();
    if (this._useWorklet && this.node) {
      const send = interleavedBuffer.slice();
      this.node.port.postMessage(
        { type: "swap", buffer: send, channels: this.channels },
        [send.buffer]
      );
    } else {
      this._spBuffer = interleavedBuffer.slice();
      this._spPosition = Math.max(
        0,
        Math.min(this.frameCount - 1, this._spPosition)
      );
    }
    this._sourceTarget = null;
    this.lufsMeasured = null;
    this._inSilence = false;
    this._measureSourceTarget();
  }
  /** Delta-add into a region of the worklet's buffer. */
  addDelta(startFrame, deltaInterleaved) {
    this._writeMirror(startFrame, deltaInterleaved, true);
    this._refreshChunks(startFrame, deltaInterleaved.length / this.channels);
    if (this._useWorklet && this.node) {
      const send = deltaInterleaved.slice();
      this.node.port.postMessage(
        { type: "add", start: startFrame, audio: send },
        [send.buffer]
      );
    } else {
      this._writeSPBuffer(startFrame, deltaInterleaved, true);
    }
  }
  setStemOverlay(kind, interleaved, channels) {
    if (!this.ctx) return;
    const state = this._stemOverlays[kind];
    state.interleaved = interleaved.slice();
    state.channels = channels;
    state.frameCount = interleaved.length / channels | 0;
    if (this._useWorklet && this.node) {
      const send = interleaved.slice();
      const node = this.node;
      node.port.postMessage(
        { type: "setOverlayBuffer", kind, buffer: send, channels },
        [send.buffer]
      );
      if (state.targetVolume > 0) {
        node.port.postMessage({
          type: "setOverlayVolume",
          kind,
          volume: state.targetVolume
        });
      }
    }
  }
  clearStemOverlays() {
    Object.keys(this._stemOverlays).forEach((kind) => {
      const state = this._stemOverlays[kind];
      state.interleaved = null;
      state.frameCount = 0;
      state.volume = 0;
      if (this._useWorklet && this.node) {
        this.node.port.postMessage({
          type: "clearOverlayBuffer",
          kind
        });
      }
    });
  }
  setStemOverlayVolume(kind, volume) {
    const state = this._stemOverlays[kind];
    const v = Math.max(0, Math.min(6, volume));
    state.targetVolume = v;
    if (this._useWorklet && this.node) {
      this.node.port.postMessage({
        type: "setOverlayVolume",
        kind,
        volume: v
      });
    }
  }
  /** Read-only view of the current buffer (for waveform rendering). */
  getMirror() {
    return this._mirror;
  }
  onMirrorChange(fn) {
    this._listeners.add(fn);
    return () => {
      this._listeners.delete(fn);
    };
  }
  async resume() {
    const ctx = this.ctx;
    if (!ctx || ctx.state !== "suspended") return;
    await new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        ctx.removeEventListener("statechange", onState);
        resolve();
      };
      const onState = () => {
        if (ctx.state !== "suspended") finish();
      };
      ctx.addEventListener("statechange", onState);
      ctx.resume().then(finish, finish);
      window.setTimeout(finish, 2e3);
    });
  }
  /**
   * Lazily create a MediaStream tee'd off the worklet output for recording.
   * Same node graph as the live destination — bit-identical to what the
   * user hears. Stays alive for the rest of the session once created.
   */
  getRecordingStream() {
    if (!this.ctx || !this.node) return null;
    if (!this._recordDest) {
      this._recordDest = this.ctx.createMediaStreamDestination();
      const tap = this._masterOut ?? this._makeupGain ?? this.node;
      tap.connect(this._recordDest);
    }
    return this._recordDest.stream;
  }
  /**
   * Toggle loudness matching. When enabled, a periodic meter tracks
   * the running-max short-term LUFS and ramps the makeup gain so
   * quieter passages match the loudest seen (peak-clamped at –1 dBTP).
   * When disabled, the meter stops, the high-water mark resets, and
   * the gain ramps back to 1.0 (mathematically transparent).
   */
  setLufs(enabled) {
    this._lufsEnabled = enabled;
    if (!this._makeupGain || !this.ctx) return;
    if (enabled) {
      if (this._mirror) {
        this._refreshChunks(0, this._mirror.length / this.channels | 0);
      }
      this._startMetering();
    } else {
      this._stopMetering();
      this.lufsMeasured = null;
      const t = this.ctx.currentTime;
      this._makeupGain.gain.cancelScheduledValues(t);
      this._makeupGain.gain.setTargetAtTime(1, t, LUFS_GAIN_RAMP_TC);
    }
  }
  /**
   * Toggle loop-at-end. When true (default), the worklet wraps the
   * playhead via the seam crossfade. When false, the playhead clamps
   * at end-of-buffer, the processor emits silence, and a one-shot
   * endOfBuffer event fires so callers can flip a paused flag.
   */
  setLoop(enabled) {
    this._loop = enabled;
    this._spEndSignaled = false;
    if (this._useWorklet && this.node) {
      this.node.port.postMessage({
        type: "setLoop",
        enabled
      });
    }
  }
  /** Jump the playhead. Seconds are clamped into the current buffer. */
  seek(positionSec) {
    if (this.frameCount === 0) return;
    const target = Math.max(
      0,
      Math.min(this.frameCount - 1, Math.round(positionSec * SAMPLE_RATE))
    );
    this.positionSec = target / SAMPLE_RATE;
    this._spEndSignaled = false;
    if (this._useWorklet && this.node) {
      this.node.port.postMessage({
        type: "seek",
        positionFrames: target
      });
    } else {
      this._spPosition = target;
    }
  }
  /** Lock playback to a sub-region of the buffer. The AudioWorklet wraps
   *  end→start each time the playhead reaches `endSec`. Pure client-side
   *  loop — the engine keeps generating linearly and writing slices into
   *  the same buffer, so what plays inside the band is whatever lives
   *  there now (regenerated audio on subsequent laps; original source
   *  in regions generation hasn't touched yet).
   *
   *  Pass min 30 ms band — anything shorter spins the wrap path tight
   *  enough to be perceived as silence. Out-of-range values are clamped.
   */
  setLoopBand(startSec, endSec) {
    if (this.frameCount === 0) return;
    const startFrames = Math.max(
      0,
      Math.min(this.frameCount - 1, Math.round(startSec * SAMPLE_RATE))
    );
    const endFrames = Math.max(
      startFrames + Math.floor(SAMPLE_RATE * 0.03),
      Math.min(this.frameCount, Math.round(endSec * SAMPLE_RATE))
    );
    if (this._useWorklet && this.node) {
      this.node.port.postMessage({
        type: "setLoopBand",
        startFrames,
        endFrames
      });
    }
  }
  /** Remove any active band loop; playback resumes wrapping at
   *  end-of-buffer (subject to `setLoop`). */
  clearLoopBand() {
    if (this._useWorklet && this.node) {
      this.node.port.postMessage({
        type: "clearLoopBand"
      });
    }
  }
  /** Subscribe to end-of-buffer hits (only fires when loop is off).
   *  Returns an unsubscribe. */
  onEndOfBuffer(fn) {
    this._endOfBufferListeners.add(fn);
    return () => {
      this._endOfBufferListeners.delete(fn);
    };
  }
  async close() {
    this._stopMetering();
    if (this._notifyTimer !== null) {
      clearTimeout(this._notifyTimer);
      this._notifyTimer = null;
    }
    this.clearStemOverlays();
    try {
      this.node?.disconnect();
    } catch {
    }
    try {
      this._masterOut?.disconnect();
    } catch {
    }
    this._recordDest = null;
    try {
      await this.ctx?.close();
    } catch {
    }
  }
  // ── internals ────────────────────────────────────────────────────────
  _startMetering() {
    if (this._meterIntervalId !== null) return;
    const audioCfg = this._loudnessConfig();
    const cfgWin = audioCfg.lufs_window_sec ?? NaN;
    this._meterWindowSec = Math.max(
      LUFS_METER_WINDOW_MIN_SEC,
      Number.isFinite(cfgWin) ? cfgWin : LUFS_METER_WINDOW_DEFAULT_SEC
    );
    this._loudnessMetric = audioCfg.lufs_metric === "dba" ? "dba" : "lufs";
    const cfgFloor = audioCfg.lufs_silence_floor_db ?? NaN;
    this._silenceFloorDb = Number.isFinite(cfgFloor) && cfgFloor > 0 ? cfgFloor : LUFS_SILENCE_FLOOR_DB_DEFAULT;
    const cfgHyst = audioCfg.lufs_silence_floor_hysteresis_db ?? NaN;
    this._silenceFloorHysteresisDb = Number.isFinite(cfgHyst) && cfgHyst >= 0 ? cfgHyst : LUFS_SILENCE_FLOOR_HYSTERESIS_DB_DEFAULT;
    this._inSilence = false;
    this._meterIntervalId = window.setInterval(
      () => this._meterTick(),
      LUFS_METER_INTERVAL_MS
    );
  }
  _stopMetering() {
    if (this._meterIntervalId === null) return;
    window.clearInterval(this._meterIntervalId);
    this._meterIntervalId = null;
  }
  /**
   * One-shot pass over the source buffer at init() / swap() that
   * captures the two numbers the matcher needs:
   *   1. integrated source loudness  -> _sourceTarget (the boost target)
   *   2. true sample peak            -> drives _peakCeiling
   *
   * Both are stable across the session; the meter loop reads them
   * but never modifies them. Reset and re-measured on swap().
   *
   * Runs synchronously on the main thread (~50-200 ms for a 60 s
   * 48 kHz buffer). Move to a worker if buffer sizes grow into the
   * multi-minute range.
   */
  _measureSourceTarget() {
    if (!this._mirror) return;
    const audioCfg = this._loudnessConfig();
    const metric = audioCfg.lufs_metric === "dba" ? "dba" : "lufs";
    this._loudnessMetric = metric;
    const { value, peak } = measureLoudness(
      this._mirror,
      this.channels,
      SAMPLE_RATE,
      metric
    );
    this._sourceTarget = value;
    const cfgHeadroom = audioCfg.lufs_peak_headroom ?? NaN;
    const headroomFactor = Number.isFinite(cfgHeadroom) ? cfgHeadroom : LUFS_PEAK_HEADROOM_DEFAULT;
    this._peakCeiling = Math.max(
      LUFS_PEAK_CEILING,
      peak * headroomFactor
    );
    const totalFrames = this._mirror.length / this.channels | 0;
    const numChunks = Math.max(1, Math.ceil(totalFrames / LUFS_CHUNK_FRAMES));
    this._chunkLoudness = new Float32Array(numChunks);
    this._chunkPeak = new Float32Array(numChunks);
    this._refreshChunks(0, totalFrames);
  }
  /**
   * Re-measure every chunk that overlaps [startFrame, startFrame+frames).
   * Cheap because chunks are 0.3 s; a 0.3 s slice typically touches
   * exactly one or two chunks. Called from patch/addDelta after the
   * mirror is updated, and from _measureSourceTarget for the full
   * initial pass.
   */
  _refreshChunks(startFrame, frames) {
    if (!this._lufsEnabled) return;
    if (!this._mirror || !this._chunkLoudness || !this._chunkPeak) return;
    const ch = this.channels;
    const totalFrames = this._mirror.length / ch | 0;
    if (totalFrames === 0 || frames <= 0) return;
    const numChunks = this._chunkLoudness.length;
    const firstChunk = Math.max(0, Math.floor(startFrame / LUFS_CHUNK_FRAMES));
    const lastChunk = Math.min(
      numChunks - 1,
      Math.floor((startFrame + frames - 1) / LUFS_CHUNK_FRAMES)
    );
    for (let c = firstChunk; c <= lastChunk; c++) {
      const cStart = c * LUFS_CHUNK_FRAMES;
      const cEnd = Math.min(cStart + LUFS_CHUNK_FRAMES, totalFrames);
      const slice = this._mirror.subarray(cStart * ch, cEnd * ch);
      const { loudness, peak } = measureBlock(slice, ch, this._loudnessMetric);
      this._chunkLoudness[c] = loudness;
      this._chunkPeak[c] = peak;
    }
  }
  _meterTick() {
    if (!this._makeupGain || !this.ctx) return;
    const target = this._sourceTarget;
    if (target === null) return;
    const map = this._chunkLoudness;
    const peakMap = this._chunkPeak;
    if (!map || !peakMap || !this._mirror) return;
    const totalFrames = this._mirror.length / this.channels | 0;
    if (totalFrames === 0) return;
    const posFramesRaw = this.positionSec * SAMPLE_RATE | 0;
    const posFrames = (posFramesRaw % totalFrames + totalFrames) % totalFrames;
    const chunkIdx = Math.min(
      map.length - 1,
      Math.floor(posFrames / LUFS_CHUNK_FRAMES)
    );
    const measured = map[chunkIdx];
    const peak = peakMap[chunkIdx];
    this.lufsMeasured = Number.isFinite(measured) ? measured : null;
    const t = this.ctx.currentTime;
    this._makeupGain.gain.cancelScheduledValues(t);
    const gap = Number.isFinite(measured) ? target - measured : Infinity;
    if (this._inSilence) {
      if (gap < this._silenceFloorDb - this._silenceFloorHysteresisDb) {
        this._inSilence = false;
      }
    } else {
      if (gap > this._silenceFloorDb) {
        this._inSilence = true;
      }
    }
    let gain;
    if (this._inSilence) {
      gain = 1;
    } else {
      const matchGain = lufsMakeupGain(measured, target, peak, this._peakCeiling);
      gain = Math.max(1, matchGain);
    }
    this._makeupGain.gain.setTargetAtTime(gain, t, LUFS_GAIN_RAMP_TC);
    if (globalThis.__LUFS_TRACE__) {
      console.log("[LUFS]", {
        positionSec: +this.positionSec.toFixed(2),
        chunk: chunkIdx,
        measured: Number.isFinite(measured) ? +measured.toFixed(2) : null,
        target: +target.toFixed(2),
        peak: +peak.toFixed(4),
        ceiling: +this._peakCeiling.toFixed(3),
        targetGain: +gain.toFixed(3),
        appliedGain: +this._makeupGain.gain.value.toFixed(3),
        disengaged: this._inSilence
      });
    }
  }
  _extractRecentWindow(seconds) {
    const mirror = this._mirror;
    if (!mirror) return null;
    const ch = this.channels;
    const totalFrames = mirror.length / ch | 0;
    const wantFrames = Math.min(
      Math.round(seconds * SAMPLE_RATE),
      totalFrames
    );
    if (wantFrames < 1) return null;
    const posFrames = (this.positionSec * SAMPLE_RATE | 0) % totalFrames;
    const startFrame = ((posFrames - wantFrames) % totalFrames + totalFrames) % totalFrames;
    const out = new Float32Array(wantFrames * ch);
    if (startFrame + wantFrames <= totalFrames) {
      out.set(
        mirror.subarray(startFrame * ch, (startFrame + wantFrames) * ch)
      );
    } else {
      const tailFrames = totalFrames - startFrame;
      out.set(mirror.subarray(startFrame * ch));
      out.set(mirror.subarray(0, (wantFrames - tailFrames) * ch), tailFrames * ch);
    }
    return out;
  }
  _writeSPBuffer(startFrame, audioInterleaved, add) {
    if (!this._spBuffer) return;
    const ch = this.channels;
    const base = startFrame * ch;
    const n = Math.min(audioInterleaved.length, this._spBuffer.length - base);
    if (n <= 0) return;
    if (add) {
      for (let i = 0; i < n; i++) this._spBuffer[base + i] += audioInterleaved[i];
    } else {
      for (let i = 0; i < n; i++) this._spBuffer[base + i] = audioInterleaved[i];
    }
  }
  _writeMirror(startFrame, audioInterleaved, add) {
    if (!this._mirror) return;
    const ch = this.channels;
    const base = startFrame * ch;
    const n = Math.min(audioInterleaved.length, this._mirror.length - base);
    if (n <= 0) return;
    if (add) {
      for (let i = 0; i < n; i++) this._mirror[base + i] += audioInterleaved[i];
    } else {
      for (let i = 0; i < n; i++) this._mirror[base + i] = audioInterleaved[i];
    }
    this._scheduleMirrorNotify();
  }
  /** Trailing-throttled mirror-change notification (~10 Hz). See the
   *  _notifyTimer comment for why per-patch synchronous fires are not
   *  affordable on the slice path. */
  _scheduleMirrorNotify() {
    if (this._notifyTimer !== null) return;
    this._notifyTimer = setTimeout(() => {
      this._notifyTimer = null;
      for (const fn of this._listeners) fn();
    }, 100);
  }
  _spProcess(e) {
    const output = e.outputBuffer;
    const frames = output.length;
    const ch = this.channels;
    const buf = this._spBuffer;
    if (!buf || this.frameCount === 0 || !this.ctx) {
      for (let c = 0; c < output.numberOfChannels; c++) {
        output.getChannelData(c).fill(0);
      }
      return;
    }
    const nFrames = this.frameCount;
    const seamFadeLen = Math.max(1, Math.floor(this.ctx.sampleRate * 0.05));
    const seam = Math.min(seamFadeLen, Math.floor(nFrames / 4));
    const outChs = [];
    for (let c = 0; c < output.numberOfChannels; c++) {
      outChs.push(output.getChannelData(c));
    }
    const overlays = [];
    {
      const ov = this._stemOverlays.vocals;
      if (ov.interleaved && ov.frameCount > 0) overlays.push(ov);
    }
    {
      const ov = this._stemOverlays.instruments;
      if (ov.interleaved && ov.frameCount > 0) overlays.push(ov);
    }
    const volAlpha = this._spOverlayVolAlpha;
    let pos = this._spPosition;
    for (let i = 0; i < frames; i++) {
      if (!this._loop && pos >= nFrames - 1) {
        for (let c = 0; c < outChs.length; c++) outChs[c][i] = 0;
        if (!this._spEndSignaled) {
          this._spEndSignaled = true;
          for (const fn of this._endOfBufferListeners) fn();
        }
        continue;
      }
      const inSeam = this._loop && seam > 0 && nFrames - pos <= seam;
      const distFromEnd = inSeam ? nFrames - pos : 0;
      const seamT = inSeam ? (seam - distFromEnd) / seam : 0;
      const headPos = inSeam ? seam - distFromEnd : 0;
      if (inSeam) {
        for (let c = 0; c < outChs.length; c++) {
          const cc = Math.min(c, ch - 1);
          const sTail = buf[pos * ch + cc];
          const sHead = buf[headPos * ch + cc];
          outChs[c][i] = sTail * (1 - seamT) + sHead * seamT;
        }
      } else {
        for (let c = 0; c < outChs.length; c++) {
          const cc = Math.min(c, ch - 1);
          outChs[c][i] = buf[pos * ch + cc];
        }
      }
      for (let oi = 0; oi < overlays.length; oi++) {
        const ov = overlays[oi];
        ov.volume += (ov.targetVolume - ov.volume) * volAlpha;
        const ovBuf = ov.interleaved;
        if (ov.volume < 1e-4) continue;
        if (pos >= ov.frameCount) continue;
        const ovCh = ov.channels;
        const ovVol = ov.volume;
        if (inSeam && headPos < ov.frameCount) {
          for (let c = 0; c < outChs.length; c++) {
            const cc = Math.min(c, ovCh - 1);
            const sTail = ovBuf[pos * ovCh + cc];
            const sHead = ovBuf[headPos * ovCh + cc];
            outChs[c][i] += (sTail * (1 - seamT) + sHead * seamT) * ovVol;
          }
        } else {
          for (let c = 0; c < outChs.length; c++) {
            const cc = Math.min(c, ovCh - 1);
            outChs[c][i] += ovBuf[pos * ovCh + cc] * ovVol;
          }
        }
      }
      pos++;
      if (pos >= nFrames) pos = this._loop ? seam : nFrames;
    }
    this._spPosition = pos;
    this.positionSec = this._spPosition / SAMPLE_RATE;
  }
};

// wsReconnect.ts
var DEFAULTS = {
  // Sized for the targeted failure modes (tunnel blip / brief network
  // drop), not for pod-level outages. Doubling base=500ms each attempt
  // with max=4s gives delays of ~0.25-0.5, 0.5-1, 1-2, 2-4, 2-4 s
  // (full-jitter), summing to a worst-case ~12s window before we hand
  // off to "refresh to retry." Long enough that one transient tunnel
  // hiccup almost always recovers; short enough that a real outage
  // doesn't leave the user staring at "Reconnecting…" wondering if the
  // app is alive. Pod death / OOM is the orchestrator's problem; we
  // shouldn't be the layer that papers over it.
  maxAttempts: 5,
  baseDelayMs: 500,
  maxDelayMs: 4e3
};
var WsReconnector = class {
  constructor(connect, handlers = {}, options = {}) {
    this.connect = connect;
    this.handlers = handlers;
    this.opts = { ...DEFAULTS, ...options };
  }
  cancelled = false;
  timer = null;
  resolveSleep = null;
  opts;
  /** Run the backoff loop. Resolves when an attempt succeeds, when the
   *  caller cancels, or when maxAttempts is reached (after invoking
   *  onGiveUp). Never throws — failures flow through the handlers. */
  async run() {
    let lastErr = new Error("no attempts ran");
    for (let attempt = 1; attempt <= this.opts.maxAttempts; attempt++) {
      if (this.cancelled) return;
      const baseMs = Math.min(
        this.opts.maxDelayMs,
        this.opts.baseDelayMs * 2 ** (attempt - 1)
      );
      const jittered = Math.round(baseMs * (0.5 + Math.random() * 0.5));
      this.handlers.onAttempt?.({
        attempt,
        maxAttempts: this.opts.maxAttempts,
        delayMs: jittered
      });
      await this.sleep(jittered);
      if (this.cancelled) return;
      try {
        await this.connect();
        if (this.cancelled) return;
        this.handlers.onSuccess?.();
        return;
      } catch (e) {
        lastErr = e instanceof Error ? e : new Error(String(e));
      }
    }
    if (!this.cancelled) this.handlers.onGiveUp?.(lastErr);
  }
  /** Cancel any in-flight backoff and stop the loop. Safe to call from
   *  any callback (useSessionStore.reset, page unload, fresh session
   *  start) — idempotent. */
  cancel() {
    this.cancelled = true;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.resolveSleep) {
      this.resolveSleep();
      this.resolveSleep = null;
    }
  }
  sleep(ms) {
    return new Promise((resolve) => {
      this.resolveSleep = resolve;
      this.timer = setTimeout(() => {
        this.resolveSleep = null;
        this.timer = null;
        resolve();
      }, ms);
    });
  }
};

// fetchWithRetry.ts
async function fetchWithRetry(url, opts = {}) {
  const deadline = Date.now() + (opts.deadlineMs ?? 18e4);
  let delay = 500;
  while (true) {
    try {
      const res = await fetch(url, { signal: opts.signal });
      if (res.ok || res.status < 500) return res;
    } catch (err2) {
      if (opts.signal?.aborted) throw err2;
    }
    if (Date.now() >= deadline) {
      throw new Error(`fetchWithRetry: gave up on ${url} after deadline`);
    }
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        opts.signal?.removeEventListener("abort", onAbort);
        resolve();
      }, Math.min(delay, 8e3));
      function onAbort() {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      }
      opts.signal?.addEventListener("abort", onAbort, { once: true });
    });
    delay = Math.min(delay * 2, 8e3);
  }
}

// fetchWireContract.ts
async function fetchWireContract(toUrl = (p) => p) {
  const res = await fetchWithRetry(toUrl("/api/protocol"));
  if (!res.ok) throw new Error(`/api/protocol failed: ${res.status}`);
  return await res.json();
}

// fetchKnobManifest.ts
async function fetchKnobManifest(sde = false, toUrl = (p) => p) {
  const res = await fetchWithRetry(toUrl(`/api/knobs${sde ? "?sde=1" : ""}`));
  if (!res.ok) throw new Error(`/api/knobs failed: ${res.status}`);
  const json = await res.json();
  return { version: json.version, knobs: json.knobs ?? {} };
}
export {
  AudioPlayer,
  COMMAND_NAMES,
  CROSSFADE_SECONDS,
  EVENT_NAMES,
  KNOB_SCHEMA_VERSION,
  PREEMPTED_CLOSE_CODE,
  PROTOCOL_VERSION,
  RemoteBackend,
  SAMPLE_RATE,
  SLICE_FLAG_DELTA,
  SLICE_FLAG_RAW,
  SLICE_HDR_SIZE,
  T,
  WsReconnector,
  fetchKnobManifest,
  fetchWireContract,
  fetchWithRetry,
  findLoudestShortTermLoudness,
  float16ArrayToFloat32,
  lufsMakeupGain,
  measureBlock,
  measureIntegratedDba,
  measureIntegratedLufs,
  measureLoudness
};
