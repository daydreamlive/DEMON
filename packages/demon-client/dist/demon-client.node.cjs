var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// node.ts
var node_exports = {};
__export(node_exports, {
  COMMAND_NAMES: () => COMMAND_NAMES,
  CONTROL_DESCRIPTIONS: () => CONTROL_DESCRIPTIONS,
  CONTROL_DISPLAY_NAMES: () => CONTROL_DISPLAY_NAMES,
  DCW_MODES: () => DCW_MODES,
  DCW_WAVELETS: () => DCW_WAVELETS,
  DEFAULT_CONFIG: () => DEFAULT_CONFIG,
  DEFAULT_TIME_SIGNATURE: () => DEFAULT_TIME_SIGNATURE,
  EVENT_NAMES: () => EVENT_NAMES,
  INTERP_PATH_DESCRIPTIONS: () => INTERP_PATH_DESCRIPTIONS,
  KNOB_SCHEMA_VERSION: () => KNOB_SCHEMA_VERSION,
  KNOWN_TOP_LEVEL_KEYS: () => KNOWN_TOP_LEVEL_KEYS,
  LOOP_GRID_ORDER: () => LOOP_GRID_ORDER,
  LORA_BLEND_DESCRIPTION: () => LORA_BLEND_DESCRIPTION,
  LORA_STRENGTH_DESCRIPTION: () => LORA_STRENGTH_DESCRIPTION,
  MANUAL_ALPHA_DESCRIPTION: () => MANUAL_ALPHA_DESCRIPTION,
  MANUAL_LAYER_DESCRIPTION: () => MANUAL_LAYER_DESCRIPTION,
  MANUAL_SRC_DESCRIPTION: () => MANUAL_SRC_DESCRIPTION,
  MANUAL_STEP_DESCRIPTION: () => MANUAL_STEP_DESCRIPTION,
  PREEMPTED_CLOSE_CODE: () => PREEMPTED_CLOSE_CODE,
  PROTOCOL_VERSION: () => PROTOCOL_VERSION,
  RCFG_MODES: () => RCFG_MODES,
  RemoteBackend: () => RemoteBackend,
  SAMPLE_RATE: () => SAMPLE_RATE,
  SERIALIZED_INPUT_KINDS: () => SERIALIZED_INPUT_KINDS,
  SLICE_FLAG_DELTA: () => SLICE_FLAG_DELTA,
  SLICE_FLAG_RAW: () => SLICE_FLAG_RAW,
  SLICE_HDR_SIZE: () => SLICE_HDR_SIZE,
  SOURCE_MODE_HINTS: () => SOURCE_MODE_HINTS,
  STEM_SECTION_DESCRIPTION: () => STEM_SECTION_DESCRIPTION,
  STEM_SOURCE_MODES: () => STEM_SOURCE_MODES,
  STRUCTURE_REF_DESCRIPTION: () => STRUCTURE_REF_DESCRIPTION,
  SWAP_SOURCE_MODES: () => SWAP_SOURCE_MODES,
  TERMS: () => TERMS,
  TIMBRE_REF_DESCRIPTION: () => TIMBRE_REF_DESCRIPTION,
  VALID_TIME_SIGNATURES: () => VALID_TIME_SIGNATURES,
  applyConfigToState: () => applyConfigToState,
  arrayBufferToBase64: () => arrayBufferToBase64,
  base64ToArrayBuffer: () => base64ToArrayBuffer,
  captureConfigFromState: () => captureConfigFromState,
  describeControl: () => describeControl,
  displayNameFor: () => displayNameFor,
  encodeWavInterleaved: () => encodeWavInterleaved,
  float16ArrayToFloat32: () => float16ArrayToFloat32,
  getUnknownKeys: () => getUnknownKeys,
  isDcwMode: () => isDcwMode,
  isDcwWavelet: () => isDcwWavelet,
  isLoopGridRes: () => isLoopGridRes,
  isRcfgMode: () => isRcfgMode,
  isSwapSourceMode: () => isSwapSourceMode,
  isTimeSignature: () => isTimeSignature,
  mergeConfig: () => mergeConfig,
  normalizeConfigShape: () => normalizeConfigShape,
  resolveControlDescription: () => resolveControlDescription,
  resolveLoraCapForSource: () => resolveLoraCapForSource,
  rtmgConfigToSessionConfig: () => rtmgConfigToSessionConfig,
  selectVariant: () => selectVariant,
  serializeConfig: () => serializeConfig,
  termFor: () => termFor,
  withUnknownKeys: () => withUnknownKeys
});
module.exports = __toCommonJS(node_exports);

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

// types/protocol.ts
var SAMPLE_RATE = 48e3;
var SLICE_HDR_SIZE = 23;
var SLICE_FLAG_RAW = 0;
var SLICE_FLAG_DELTA = 1;
var PREEMPTED_CLOSE_CODE = 4001;

// protocol.ts
var import_meta = {};
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
  // WebSocket implementation + its OPEN ready-state constant. Injected so
  // non-browser hosts (Node-for-Max) supply `ws` and never touch a global
  // `WebSocket`. OPEN is fixed at 1 by the WHATWG spec across every
  // implementation, so the fallback is exact when nothing is resolvable.
  _wsCtor;
  _wsOpen;
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
    this._wsCtor = opts.WebSocketConstructor;
    this._wsOpen = (opts.WebSocketConstructor ?? (typeof WebSocket !== "undefined" ? WebSocket : void 0))?.OPEN ?? 1;
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
        new URL("./workers/sliceDecoder.worker.ts", import_meta.url),
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
      const WS = this._wsCtor ?? WebSocket;
      const ws = new WS(this.url);
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
            if (typeof msg.source_epoch === "number")
              this._sliceEpoch = msg.source_epoch;
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
          this._sliceEpoch = meta.source_epoch ?? this._sliceEpoch + 1;
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
            case "audio_written":
              this.dispatchEvent(
                new CustomEvent("audio_written", { detail: msg })
              );
              break;
            case "audio_write_failed":
              this.dispatchEvent(
                new CustomEvent("audio_write_failed", { detail: msg.error })
              );
              break;
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
    if (this.ws?.readyState !== this._wsOpen) return false;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
   * Change live repaint/extend/cover state. This does not launch a one-shot
   * job: new requests enter the existing ring and emerge through ordinary
   * windowed slices. Disable with `enabled: false` (regions may be empty).
   */
  sendSetAudioEdit(regions, options = {}) {
    if (this.ws?.readyState !== this._wsOpen) return;
    try {
      const msg = {
        type: "set_audio_edit",
        enabled: options.enabled ?? true,
        regions,
        source_mode: options.sourceMode ?? "waveform",
        strength: Math.max(0, Math.min(1, options.strength ?? 1))
      };
      this.ws.send(JSON.stringify(msg));
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return;
    try {
      const msg = { type: "disable_lora", id };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /** Add the next manual steering slot (LIFO). Server echoes
   *  ``manual_slot_count`` on success or refusal. */
  sendManualSlotAdd() {
    if (this.ws?.readyState !== this._wsOpen) return;
    try {
      const msg = { type: "manual_slot_add" };
      this.ws.send(JSON.stringify(msg));
    } catch {
    }
  }
  /** Pop the highest-numbered manual steering slot. */
  sendManualSlotPop() {
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return false;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return;
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
    if (this.ws?.readyState !== this._wsOpen) return false;
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
  /**
   * Write audio onto the live source in place — the "play into the model"
   * path, with NO song restart / playhead reset / BPM-key re-detect. The
   * binary PCM frame is ONLY the audio being written (a bar, a chunk, or a
   * whole period): the server keeps a sample-exact mirror of the source and
   * pulls all re-encode context from it. The single input primitive behind
   * the M4L feed modes (stream / splice / print / autosplice) and the VST's
   * live splice; the browser app drives generation, not source writes, so
   * it has no caller yet.
   *
   * `opts.sourceEpoch` pins the write to the source generation it was
   * computed against (from `ready`/`swap_ready`, i.e. this client's
   * `sliceEpoch`); a server-side mismatch is rejected with
   * `audio_write_failed` rather than splicing into the wrong source.
   * `opts.dropIfBusy` skips the send while the WS send buffer is still
   * draining (>1 MiB) — set it for streamed tape-head writes so a slow
   * link drops stale frames instead of queueing an ever-staler backlog;
   * never set it for a one-shot commit that must land. Acked by
   * `audio_written` / `audio_write_failed`.
   */
  sendWriteAudio(interleaved, channels, opts = {}) {
    if (this.ws?.readyState !== this._wsOpen) return false;
    if (opts.dropIfBusy && (this.ws.bufferedAmount || 0) > 1 << 20) return false;
    try {
      const msg = { type: "write_audio" };
      if (opts.atS != null) msg.at_s = opts.atS;
      if (opts.mix) msg.mix = opts.mix;
      if (opts.repeat) msg.repeat = opts.repeat;
      if (opts.sourceEpoch != null) msg.source_epoch = opts.sourceEpoch;
      if (opts.refreshTimbre) msg.refresh_timbre = true;
      this.ws.send(JSON.stringify(msg));
      this.ws.send(packPcmFrame(interleaved, channels));
      return true;
    } catch (e) {
      console.error("[protocol] write_audio failed:", e);
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
  /** Current source-buffer epoch (0 at create, bumped by every swap) — the
   *  client mirror of the server's `source_epoch`. Read it to tag a slice
   *  consumer's local buffer generation and to pin `write_audio` sends to
   *  the live source. */
  get sliceEpoch() {
    return this._sliceEpoch;
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

// types/wireContract.gen.ts
var PROTOCOL_VERSION = 1;
var KNOB_SCHEMA_VERSION = 1;
var COMMAND_NAMES = [
  "params",
  "loop_band",
  "prompt",
  "set_audio_edit",
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
  "audio_edit_applied",
  "audio_edit_failed",
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

// config/enums.ts
var LOOP_GRID_ORDER = ["bar", "half", "beat", "eighth"];
function isLoopGridRes(v) {
  return typeof v === "string" && LOOP_GRID_ORDER.includes(v);
}
var DCW_MODES = ["low", "high", "double", "pix"];
var DCW_WAVELETS = ["haar", "db4", "sym8", "db8"];
function isDcwMode(v) {
  return typeof v === "string" && DCW_MODES.includes(v);
}
function isDcwWavelet(v) {
  return typeof v === "string" && DCW_WAVELETS.includes(v);
}
var RCFG_MODES = ["off", "initialize", "self"];
function isRcfgMode(v) {
  return typeof v === "string" && RCFG_MODES.includes(v);
}
var VALID_TIME_SIGNATURES = ["2", "3", "4", "6"];
var DEFAULT_TIME_SIGNATURE = "4";
function isTimeSignature(v) {
  return typeof v === "string" && VALID_TIME_SIGNATURES.includes(v);
}

// config/types.ts
var SWAP_SOURCE_MODES = ["full", "vocals", "instruments"];

// config/defaults.ts
var DEFAULT_CONFIG = {
  version: 1,
  engine: {
    sde: false,
    lora: true,
    depth: 4,
    vae_window: 0.36,
    crop: 0,
    steps: 8,
    fast_vae: false,
    walk_window: false,
    walk_window_s: 60,
    lead_floor_s: 0.25,
    lead_ceiling_s: 1.35,
    lead_release_tau_s: 1.5,
    max_source_duration_s: 120,
    key: "G# minor",
    time_signature: DEFAULT_TIME_SIGNATURE,
    enabled_loras: [],
    auto_prepend_lora_triggers: true,
    show_incompatible_loras: false
  },
  prompts: {
    a: "heavy dubstep, deathstep, afxdump, growl heavy bass distortion",
    b: "daft punk style, beautiful, four to the floor, angelic",
    blend: 0.4
  },
  controls: {
    denoise: 0.7,
    hint_strength: 1,
    feedback: 0,
    feedback_depth: 1,
    shift: 3.5,
    ch_g0: 1,
    ch_g1: 1,
    ch_g2: 1,
    ch_g3: 1,
    ch_g4: 1,
    ch_g5: 1,
    ch_g6: 1,
    ch_g7: 1,
    ch13: 1,
    ch14: 1,
    ch19: 1,
    ch23: 1,
    ch29: 1,
    ch56: 1,
    dcw_scaler: 0.05,
    dcw_high_scaler: 0.02,
    dcw_enabled: true,
    dcw_mode: "double",
    dcw_wavelet: "haar",
    lora_default_strength: 1.4,
    guidance_scale: 7,
    cfg_rescale: 0,
    rcfg_mode: "off"
  },
  channel_ranges: {
    ch_g0: { min: 0, max: 2.2, reverse: false },
    ch_g1: { min: 0, max: 2, reverse: false },
    ch_g2: { min: 0, max: 2.3, reverse: true },
    ch_g3: { min: 0, max: 2, reverse: false },
    ch_g4: { min: 0, max: 2.5, reverse: false },
    ch_g5: { min: 0, max: 2, reverse: false },
    ch_g6: { min: 0, max: 2, reverse: true },
    ch_g7: { min: 0, max: 2, reverse: true },
    ch13: { min: 0, max: 2, reverse: true },
    ch14: { min: 0, max: 2.3, reverse: false },
    ch19: { min: 0, max: 2.5, reverse: false },
    ch23: { min: 0, max: 2.45, reverse: false },
    ch29: { min: 0, max: 2, reverse: false },
    ch56: { min: 0, max: 2, reverse: false }
  },
  seed: 0,
  swap_source_mode: "instruments",
  web: {
    effects: {
      parallax_strength: 0.4,
      bloom_on_kick: 0.3,
      bloom_threshold: 0.15,
      warp_strength: 0.4
    },
    audio: {
      lufs_enabled: false,
      lufs_window_sec: 3,
      lufs_metric: "lufs",
      lufs_peak_headroom: 4,
      lufs_silence_floor_db: 30,
      lufs_silence_floor_hysteresis_db: 6
    },
    reset_seconds: 0,
    denoise_session_gate: {
      enabled: true,
      glide_ms: 700
    },
    restart_song_on_swap: true
  }
};

// config/transforms.ts
function isSwapSourceMode(v) {
  return SWAP_SOURCE_MODES.includes(v);
}
var KNOWN_TOP_LEVEL_KEYS = /* @__PURE__ */ new Set([
  "version",
  "engine",
  "prompts",
  "controls",
  "channel_ranges",
  "seed",
  "swap_source_mode",
  "web",
  "curves",
  "loop"
]);
var UNKNOWN = Symbol("rtmgConfigUnknownKeys");
function getUnknownKeys(cfg) {
  return cfg[UNKNOWN] ?? {};
}
function withUnknownKeys(cfg, unknown) {
  if (Object.keys(unknown).length === 0) return cfg;
  Object.defineProperty(cfg, UNKNOWN, {
    value: unknown,
    enumerable: false,
    configurable: true,
    writable: true
  });
  return cfg;
}
function serializeConfig(cfg) {
  return { ...getUnknownKeys(cfg), ...cfg };
}
var LEGACY_WEB_KEYS = [
  "effects",
  "audio",
  "reset_seconds",
  "denoise_session_gate",
  "restart_song_on_swap"
];
function normalizeConfigShape(raw) {
  const hasLegacyTopLevel = LEGACY_WEB_KEYS.some((k) => k in raw);
  if (!hasLegacyTopLevel) return raw;
  const liftedWeb = { ...raw.web ?? {} };
  for (const k of LEGACY_WEB_KEYS) {
    if (k in raw && !(k in liftedWeb))
      liftedWeb[k] = raw[k];
  }
  const out = { ...raw, web: liftedWeb };
  for (const k of LEGACY_WEB_KEYS) delete out[k];
  return out;
}
function mergeConfig(base, rawOverride) {
  if (!rawOverride) return base;
  const override = normalizeConfigShape(
    rawOverride
  );
  const web = override.web ?? {};
  const merged = {
    version: typeof override.version === "number" ? override.version : base.version,
    engine: { ...base.engine, ...override.engine ?? {} },
    prompts: { ...base.prompts, ...override.prompts ?? {} },
    controls: { ...base.controls, ...override.controls ?? {} },
    // Per-param shallow merge: an override entry replaces the matching
    // base entry whole (operator-supplied {min,max,reverse} must travel
    // together to be coherent). Unspecified params keep the bundled
    // default range.
    channel_ranges: {
      ...base.channel_ranges,
      ...override.channel_ranges ?? {}
    },
    seed: typeof override.seed === "number" ? override.seed : base.seed,
    swap_source_mode: isSwapSourceMode(override.swap_source_mode) ? override.swap_source_mode : base.swap_source_mode,
    web: {
      effects: { ...base.web.effects, ...web.effects ?? {} },
      audio: { ...base.web.audio, ...web.audio ?? {} },
      reset_seconds: typeof web.reset_seconds === "number" ? web.reset_seconds : base.web.reset_seconds,
      denoise_session_gate: {
        ...base.web.denoise_session_gate,
        ...web.denoise_session_gate ?? {}
      },
      restart_song_on_swap: typeof web.restart_song_on_swap === "boolean" ? web.restart_song_on_swap : base.web.restart_song_on_swap
    },
    // Curves are operator-authored and only meaningful as a whole bag,
    // so the override entry replaces the base entry whole when present.
    // Absent override keeps whatever the base has (DEFAULT_CONFIG leaves
    // this undefined; stock pods fall through to localStorage hydration).
    ...override.curves !== void 0 ? { curves: override.curves } : base.curves !== void 0 ? { curves: base.curves } : {},
    // Loop region replaces whole when the import carries one; otherwise
    // keep whatever the base (the live config) holds.
    ...override.loop !== void 0 ? { loop: override.loop } : base.loop !== void 0 ? { loop: base.loop } : {}
  };
  const unknown = { ...getUnknownKeys(base) };
  for (const k of Object.keys(override)) {
    if (!KNOWN_TOP_LEVEL_KEYS.has(k)) {
      unknown[k] = override[k];
    }
  }
  return withUnknownKeys(merged, unknown);
}
function resolveLoraCapForSource(durationS, engine) {
  const tiers = engine.max_concurrent_loras_tiers;
  if (tiers && tiers.length > 0) {
    const sorted = [...tiers].filter(
      (t) => typeof t?.up_to_s === "number" && typeof t?.cap === "number"
    ).sort((a, b) => a.up_to_s - b.up_to_s);
    for (const tier of sorted) {
      if (durationS <= tier.up_to_s) return tier.cap;
    }
  }
  const fallback = engine.max_concurrent_loras;
  return typeof fallback === "number" && fallback > 0 ? fallback : null;
}
function selectVariant(cfg, scale) {
  if (scale !== "5B") return cfg;
  const e = cfg.engine;
  const p = cfg.prompts;
  return withUnknownKeys(
    {
      ...cfg,
      engine: {
        ...e,
        depth: e.depth_xl ?? e.depth,
        enabled_loras: e.enabled_loras_xl ?? e.enabled_loras
      },
      prompts: {
        ...p,
        a: p.a_xl ?? p.a,
        b: p.b_xl ?? p.b,
        blend: typeof p.blend_xl === "number" ? p.blend_xl : p.blend
      }
    },
    getUnknownKeys(cfg)
  );
}

// config/wire.ts
function rtmgConfigToSessionConfig(cfg, runtime) {
  const e = cfg.engine;
  return {
    telemetry_version: 1,
    sde: e.sde,
    lora: e.lora,
    depth: e.depth,
    vae_window: e.vae_window,
    crop: e.crop,
    steps: e.steps,
    fast_vae: e.fast_vae,
    walk_window: e.walk_window ?? false,
    walk_window_s: e.walk_window_s ?? 60,
    lead_floor_s: e.lead_floor_s,
    lead_ceiling_s: e.lead_ceiling_s,
    lead_release_tau_s: e.lead_release_tau_s,
    enabled_loras: runtime.enabledLoras,
    prompt: runtime.promptA,
    prompt_b: runtime.promptB,
    lora_strengths: runtime.loraStrengths,
    fixture_name: runtime.fixtureName,
    ...runtime.stemSourceMode ? { stem_source_mode: runtime.stemSourceMode } : {},
    use_server_fixture: runtime.useServerFixture,
    ...runtime.clientId ? { client_id: runtime.clientId } : {}
  };
}
function applyConfigToState(cfg, scale) {
  const resolved = selectVariant(cfg, scale);
  const sliderUpdates = {};
  for (const [k, v] of Object.entries(resolved.controls)) {
    if (typeof v === "number") sliderUpdates[k] = v;
  }
  sliderUpdates.prompt_blend = resolved.prompts.blend;
  const patch = {
    resolved,
    sliderUpdates,
    promptA: resolved.prompts.a,
    promptB: resolved.prompts.b,
    activeKey: resolved.engine.key,
    activeTimeSignature: isTimeSignature(resolved.engine.time_signature) ? resolved.engine.time_signature : DEFAULT_TIME_SIGNATURE,
    seed: resolved.seed,
    lufsOn: resolved.web.audio.lufs_enabled
  };
  if (typeof resolved.controls.dcw_enabled === "boolean") {
    patch.dcwEnabled = resolved.controls.dcw_enabled;
  }
  if (isDcwMode(resolved.controls.dcw_mode)) {
    patch.dcwMode = resolved.controls.dcw_mode;
  }
  if (isDcwWavelet(resolved.controls.dcw_wavelet)) {
    patch.dcwWavelet = resolved.controls.dcw_wavelet;
  }
  if (isRcfgMode(resolved.controls.rcfg_mode)) {
    patch.rcfgMode = resolved.controls.rcfg_mode;
  }
  if (resolved.loop) {
    const lp = resolved.loop;
    const validBand = lp.band && typeof lp.band.start === "number" && typeof lp.band.end === "number" && lp.band.end > lp.band.start ? { start: lp.band.start, end: lp.band.end } : null;
    patch.loop = {
      loopBand: validBand,
      bandLoopEnabled: typeof lp.enabled === "boolean" ? lp.enabled : true,
      loopGridRes: isLoopGridRes(lp.grid) ? lp.grid : "beat"
    };
    if (typeof lp.fullBuffer === "boolean") patch.loop.loopOn = lp.fullBuffer;
  }
  if (resolved.curves) {
    patch.curves = {
      scheduleEnabled: resolved.curves.scheduleEnabled,
      curves: Object.fromEntries(
        Object.entries(resolved.curves.curves).map(([param, curve]) => [
          param,
          {
            enabled: curve.enabled,
            points: curve.points.map((p) => ({ x: p.x, y: p.y, mode: p.mode }))
          }
        ])
      )
    };
  }
  return patch;
}
function captureConfigFromState(snapshot, base) {
  return withUnknownKeys(
    {
      version: base.version,
      engine: {
        ...base.engine,
        key: snapshot.key,
        time_signature: snapshot.timeSignature,
        enabled_loras: snapshot.enabledLoras
      },
      prompts: {
        a: snapshot.promptA,
        b: snapshot.promptB,
        blend: snapshot.promptBlend
      },
      controls: snapshot.controls,
      channel_ranges: base.channel_ranges,
      seed: snapshot.seed,
      swap_source_mode: base.swap_source_mode,
      web: {
        effects: base.web.effects,
        audio: { ...base.web.audio, lufs_enabled: snapshot.lufsOn },
        reset_seconds: base.web.reset_seconds,
        denoise_session_gate: base.web.denoise_session_gate,
        restart_song_on_swap: base.web.restart_song_on_swap
      },
      loop: snapshot.loop,
      curves: snapshot.curves
    },
    getUnknownKeys(base)
  );
}

// inputs.ts
var STEM_SOURCE_MODES = ["full", "vocals", "instruments"];
var SERIALIZED_INPUT_KINDS = ["fixture", "clip"];
function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}
function encodeWavInterleaved(interleaved, channels, sampleRate) {
  const frames = Math.floor(interleaved.length / channels);
  const bytesPerSample = 2;
  const dataLen = frames * channels * bytesPerSample;
  const out = new ArrayBuffer(44 + dataLen);
  const view = new DataView(out);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLen, true);
  const pcm = new Int16Array(out, 44, frames * channels);
  const n = frames * channels;
  for (let i = 0; i < n; i++) {
    const s = interleaved[i];
    const c = s < -1 ? -1 : s > 1 ? 1 : s;
    pcm[i] = c < 0 ? c * 32768 : c * 32767;
  }
  return out;
}
var B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
var B64_LOOKUP = (() => {
  const t = new Int16Array(256).fill(-1);
  for (let i = 0; i < B64.length; i++) t[B64.charCodeAt(i)] = i;
  return t;
})();
function arrayBufferToBase64(buf) {
  const bytes = new Uint8Array(buf);
  const len = bytes.length;
  let out = "";
  let i = 0;
  for (; i + 2 < len; i += 3) {
    const n = bytes[i] << 16 | bytes[i + 1] << 8 | bytes[i + 2];
    out += B64[n >> 18 & 63] + B64[n >> 12 & 63] + B64[n >> 6 & 63] + B64[n & 63];
  }
  const rem = len - i;
  if (rem === 1) {
    const n = bytes[i] << 16;
    out += B64[n >> 18 & 63] + B64[n >> 12 & 63] + "==";
  } else if (rem === 2) {
    const n = bytes[i] << 16 | bytes[i + 1] << 8;
    out += B64[n >> 18 & 63] + B64[n >> 12 & 63] + B64[n >> 6 & 63] + "=";
  }
  return out;
}
function base64ToArrayBuffer(b64) {
  const bytes = new Uint8Array(Math.floor(b64.length * 3 / 4));
  let bi = 0;
  let acc = 0;
  let accBits = 0;
  for (let i = 0; i < b64.length; i++) {
    const v = B64_LOOKUP[b64.charCodeAt(i)];
    if (v < 0) continue;
    acc = acc << 6 | v;
    accBits += 6;
    if (accBits >= 8) {
      accBits -= 8;
      bytes[bi++] = acc >> accBits & 255;
    }
  }
  return bytes.buffer.slice(0, bi);
}

// controls/copy.ts
var CONTROL_DISPLAY_NAMES = {
  // Macros where the friendly name reads more clearly than the
  // engine-honest one: `strength` for `denoise` (the "denoise"
  // engine-internal naming refers to the diffusion step, but users
  // perceive this knob as "how strong is the remix"), `structure` for
  // `hint_strength`, `timbre` for `timbre_strength` (drop the
  // "strength" suffix on knobs — the value readout already conveys
  // magnitude). Everything else falls back to displayNameFor
  // (underscore → space) so the UI matches what the engine, MIDI map,
  // and config files call them.
  denoise: "strength",
  hint_strength: "structure",
  timbre_strength: "timbre",
  // dcw_* keep their engine-honest "DCW low" / "DCW high" — these are
  // DCW-internal scalers, not generic EQ.
  dcw_scaler: "DCW low",
  dcw_high_scaler: "DCW high"
};
var CONTROL_DESCRIPTIONS = {
  // ── Main remix controls ──
  denoise: "How much the model reshapes the source audio. Keep it low for a subtle remix that stays close to the original; push it high to fully transform the track into something new. The most expressive knob \u2014 try sweeping it during playback.",
  hint_strength: "How closely the model follows the original song's structure \u2014 sections, rhythm, dynamics. Crank it up to keep the arrangement intact; drop it to let the model rearrange more freely.",
  timbre_strength: "How much of the source's instrument character (tone, color) carries into the output. High keeps the original instruments recognizable; low frees the model to swap them for whatever fits the prompt.",
  // ── Engine internals ──
  feedback: "How similar each new generation is to the previous one. Low values give you variety on every refresh; higher values give you a continuous evolution where each generation flows into the next. 0.3\u20130.5 is the sweet spot for smooth continuity without everything sounding the same.",
  feedback_depth: "How far back in time the Feedback knob reaches. 1 (default) blends with the most recent generation. Higher values reach back several ticks for an echo / ghost effect \u2014 a faint repeat of an earlier moment surfaces in the current output. Lets you get distant feedback without cranking Feedback all the way up.",
  shift: "Advanced: changes where the model concentrates its work across denoising. The default is tuned for the turbo engine and works well in most cases \u2014 leave it alone unless you're chasing a specific feel.",
  steps_override: "Diffusion step count. Lower steps = lower quality. Higher steps = more latency. Default 8 is the turbo balance. Changing this rebuilds the streaming pipeline, so expect a brief audio glitch when you move it.",
  guidance_scale: "CFG strength. Only takes effect when the RCFG mode dropdown below is NOT 'off'. Higher values push the output further toward the prompt at the cost of more artifacts. Turbo is CFG-distilled, so the useful range is narrower than a base SD model \u2014 try 3\u20138.",
  cfg_rescale: "After CFG, mix the guided velocity's magnitude back toward what the positive forward produced. 0 keeps raw CFG; 1 fully snaps the magnitude. Pair with high guidance_scale to keep the prompt-push without the harshness that high CFG causes on its own.",
  // ── Activation steering (auto path) ──
  // Each tooltip names the underlying probe cell so the operator can
  // recreate the effect on a manual slot.
  steer_bright: "Activation-steering: positive alpha shifts spectral centroid up (brighter, more highs). 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector brightness_l09_t3 at layer = 9, step = round(3/8 x steps_count).",
  steer_warm: "Activation-steering: positive alpha tilts the spectrum toward bass (warmer). The raw vector points the wrong way for this axis, so this knob folds in a -1 sign. 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector warmth_l15_t0 at layer = 15, step = 0, then INVERT alpha sign (manual mode is sign-agnostic).",
  steer_rough: "Activation-steering: positive alpha increases spectral flatness (grittier, noisier). Vector magnitude at this probe cell is small, so effect builds slowly. 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector roughness_l09_t3 at layer = 9, step = round(3/8 x steps_count).",
  steer_density: "Activation-steering: positive alpha thins the texture toward sparse/minimal. Inject layer is shifted 3 shallower than the probe layer (Phase-3 transfer finding). 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector density_l18_t3 at layer = 15 (probe 18 minus 3), step = round(3/8 x steps_count).",
  // ── DCW ──
  dcw_scaler: "Experimental \u2014 adjusts the low-band strength of an internal correction the model applies to itself during generation (DCW). This scaler is active in the early part of the run. The exact audio mapping is still being explored \u2014 sweep it to discover what it does for your source. Extreme values can be unpredictable but cool.",
  dcw_high_scaler: "Experimental \u2014 adjusts the high-band strength of an internal correction the model applies to itself during generation (DCW). This scaler is active in the later part of the run. The exact audio mapping is still being explored \u2014 sweep it to discover what it does for your source. Extreme values can be unpredictable but cool."
};
var CHANNEL_GAINS = ["ch_g0", "ch_g1", "ch_g2", "ch_g3", "ch_g4", "ch_g5", "ch_g6", "ch_g7"];
var NAMED_CHANNELS = ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"];
for (const [i, p] of CHANNEL_GAINS.entries()) {
  CONTROL_DESCRIPTIONS[p] = `Experimental \u2014 adjusts the strength of one of the model's internal latent channels (channel ${i}). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.`;
}
for (const p of NAMED_CHANNELS) {
  const idx = p.slice(2);
  CONTROL_DESCRIPTIONS[p] = `Experimental \u2014 a hand-picked internal latent channel (#${idx}) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source.`;
}
var LORA_STRENGTH_DESCRIPTION = "How strongly this LoRA shapes the output. LoRAs are little style packs \u2014 set a low value for a subtle flavor, crank past 1.0 to make this LoRA dominate the sound. Multiple LoRAs stack \u2014 turn several on at once for combined styles.";
var LORA_BLEND_DESCRIPTION = "Crossfade between LoRA A and LoRA B. 0 = A only, 1 = B only, 0.5 = both at half strength. Use this to morph between two styles smoothly.";
var MANUAL_SRC_DESCRIPTION = "Catalog index of the steering vector this slot fires. The catalog enumerates every pre-built (axis, build_layer, build_step) cell on disk in stable axis-major order. Double-click the readout to type an exact index; query the MCP list_manual_steering_vectors tool for the full table. Has no effect until \u03B1 is non-zero.";
var MANUAL_LAYER_DESCRIPTION = "DiT inject layer (0-23). The vector is added to this layer's post-block residual. Bypasses the auto path's density layer offset \u2014 the value lands exactly where you point it.";
var MANUAL_STEP_DESCRIPTION = "Diffusion inject step (0-15). Bypasses the auto path's fractional step mapping; the engine fires the injection only on the step that matches this value. If you pick a step past the current steps count - 1, the slot stays silent until you raise the step count.";
var MANUAL_ALPHA_DESCRIPTION = "Strength of this manual slot's injection. 0 disables the slot. Negative \u03B1 inverts the vector's direction at injection time (no sign correction is applied; what you set is what the engine receives). Sweep range and breakage point mirror the perceptual steering knobs.";
var INTERP_PATH_DESCRIPTIONS = {
  structure: "How structural (semantic-hint) guidance blends in. Slerp holds the latent's norm constant; linear averages.",
  timbre: "How the timbre reference blends from silence to full. Slerp holds the conditioning norm constant; linear averages.",
  prompt: "How prompt A crossfades to prompt B. Slerp avoids the washed-out midpoint a linear average produces between unrelated prompts.",
  feedback: "How the latent feedback tap mixes into the source. Slerp holds the latent's norm constant; linear averages."
};
var SOURCE_MODE_HINTS = {
  full: "Feed the whole mix to inference",
  instruments: "Feed only the instrumental bed to inference",
  vocals: "Feed only the vocal stem to inference"
};
var TIMBRE_REF_DESCRIPTION = "Optional reference audio for the timbre channel. Picking a track here biases the model's instrument character toward what's in that file, leaving structure (rhythm, sections) free to follow the playing input. Default 'Input Track' uses the playing source's own latent.";
var STRUCTURE_REF_DESCRIPTION = "Optional reference audio for the structure channel. Picking a track here biases the model's section/rhythm/dynamics layout toward that file, leaving timbre (instrument character) free to follow the playing input. Default 'Input Track' uses the playing source's own latent.";
var STEM_SECTION_DESCRIPTION = "Vocal and instrumental stems extracted from the source track. Drag a panner right to mix that layer into the model output. Click the layer name to mute or unmute without losing the level. Hold V (vocals) or I (instruments) + \u25B2\u25BC to nudge from the keyboard.";

// controls/lexicon.ts
var TERMS = {
  /** The prompt/tags input feature. */
  tags: "Tags",
  /** A LoRA, singular — for compound labels like `${TERMS.lora} Library`. */
  lora: "LoRA",
  /** LoRAs, plural — for counts, placeholders, empty states. */
  lora_plural: "LoRAs"
};
function termFor(id) {
  return TERMS[id];
}

// controls/index.ts
function displayNameFor(param) {
  return CONTROL_DISPLAY_NAMES[param] ?? param.replace(/_/g, " ");
}
function describeControl(param) {
  if (param.startsWith("lora_str_")) return LORA_STRENGTH_DESCRIPTION;
  if (param === "lora_blend") return LORA_BLEND_DESCRIPTION;
  if (param.startsWith("man_src_")) return MANUAL_SRC_DESCRIPTION;
  if (param.startsWith("man_layer_")) return MANUAL_LAYER_DESCRIPTION;
  if (param.startsWith("man_step_")) return MANUAL_STEP_DESCRIPTION;
  if (param.startsWith("man_alpha_")) return MANUAL_ALPHA_DESCRIPTION;
  return CONTROL_DESCRIPTIONS[param];
}
function resolveControlDescription(param, manifest) {
  return describeControl(param) ?? manifest?.[param]?.description;
}
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  COMMAND_NAMES,
  CONTROL_DESCRIPTIONS,
  CONTROL_DISPLAY_NAMES,
  DCW_MODES,
  DCW_WAVELETS,
  DEFAULT_CONFIG,
  DEFAULT_TIME_SIGNATURE,
  EVENT_NAMES,
  INTERP_PATH_DESCRIPTIONS,
  KNOB_SCHEMA_VERSION,
  KNOWN_TOP_LEVEL_KEYS,
  LOOP_GRID_ORDER,
  LORA_BLEND_DESCRIPTION,
  LORA_STRENGTH_DESCRIPTION,
  MANUAL_ALPHA_DESCRIPTION,
  MANUAL_LAYER_DESCRIPTION,
  MANUAL_SRC_DESCRIPTION,
  MANUAL_STEP_DESCRIPTION,
  PREEMPTED_CLOSE_CODE,
  PROTOCOL_VERSION,
  RCFG_MODES,
  RemoteBackend,
  SAMPLE_RATE,
  SERIALIZED_INPUT_KINDS,
  SLICE_FLAG_DELTA,
  SLICE_FLAG_RAW,
  SLICE_HDR_SIZE,
  SOURCE_MODE_HINTS,
  STEM_SECTION_DESCRIPTION,
  STEM_SOURCE_MODES,
  STRUCTURE_REF_DESCRIPTION,
  SWAP_SOURCE_MODES,
  TERMS,
  TIMBRE_REF_DESCRIPTION,
  VALID_TIME_SIGNATURES,
  applyConfigToState,
  arrayBufferToBase64,
  base64ToArrayBuffer,
  captureConfigFromState,
  describeControl,
  displayNameFor,
  encodeWavInterleaved,
  float16ArrayToFloat32,
  getUnknownKeys,
  isDcwMode,
  isDcwWavelet,
  isLoopGridRes,
  isRcfgMode,
  isSwapSourceMode,
  isTimeSignature,
  mergeConfig,
  normalizeConfigShape,
  resolveControlDescription,
  resolveLoraCapForSource,
  rtmgConfigToSessionConfig,
  selectVariant,
  serializeConfig,
  termFor,
  withUnknownKeys
});
