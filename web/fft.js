// FFT and quantized radial lookup from anttiluode/SighImageInverse (MIT).
const KNOTS=10;
  function fft1(re, im, inverse) {
    var n = re.length, j = 0;
    for (var i = 1; i < n; i++) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        var t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (var len = 2; len <= n; len <<= 1) {
      var angle = (inverse ? 2 : -2) * Math.PI / len;
      var wr0 = Math.cos(angle), wi0 = Math.sin(angle);
      for (var off = 0; off < n; off += len) {
        var wr = 1, wi = 0;
        for (var k = 0; k < len / 2; k++) {
          var a = off + k, b = a + len / 2;
          var vr = re[b] * wr - im[b] * wi;
          var vi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - vr; im[b] = im[a] - vi;
          re[a] += vr; im[a] += vi;
          var next = wr * wr0 - wi * wi0;
          wi = wr * wi0 + wi * wr0;
          wr = next;
        }
      }
    }
    if (inverse) for (var p = 0; p < n; p++) { re[p] /= n; im[p] /= n; }
  }
  function fft2(array, N, inverse) {
    if ((N & (N - 1)) || array.length !== N * N) throw Error("Square power-of-two image required");
    var re = new Float64Array(array), im = new Float64Array(N * N);
    transform2(re, im, N, !!inverse);
    return { re: re, im: im };
  }
  function transform2(re, im, N, inverse) {
    var a = new Float64Array(N), b = new Float64Array(N);
    for (var y = 0; y < N; y++) {
      for (var x = 0; x < N; x++) { a[x] = re[y * N + x]; b[x] = im[y * N + x]; }
      fft1(a, b, inverse);
      for (var xx = 0; xx < N; xx++) { re[y * N + xx] = a[xx]; im[y * N + xx] = b[xx]; }
    }
    for (var x2 = 0; x2 < N; x2++) {
      for (var yy = 0; yy < N; yy++) { a[yy] = re[yy * N + x2]; b[yy] = im[yy * N + x2]; }
      fft1(a, b, inverse);
      for (var y2 = 0; y2 < N; y2++) { re[y2 * N + x2] = a[y2]; im[y2 * N + x2] = b[y2]; }
    }
  }
  function lookupIndex(y, x, N) {
    var ky = y <= N / 2 ? y : y - N;
    var kx = x <= N / 2 ? x : x - N;
    // k^2.max = (N/2)^2 + (N/2)^2 on even square grids
    return Math.min(255, Math.floor(255 * (ky * ky + kx * kx) / (N * N / 2)));
  }
  function basis(bin) {
    // Same 256-entry curve and quantized floor indexing as the Python Sigh source.
    var u = bin / 255 * (KNOTS - 1);
    var i = Math.min(KNOTS - 2, Math.floor(u));
    return [i, 1 - (u - i), u - i];
  }
  function gainAt(bin, gains) {
    var b = basis(bin);
    return b[1] * gains[b[0]] + b[2] * gains[b[0] + 1];
  }
  function fromSpectrum(fr, fi, N) {
    var re = new Float64Array(fr), im = new Float64Array(fi);
    transform2(re, im, N, true);
    return re;
  }
