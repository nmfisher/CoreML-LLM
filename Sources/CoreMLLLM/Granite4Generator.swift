// IBM Granite 4.1 3B (and other Granite-style dense GQA decoders) —
// text-only stateful chunked generator.
//
// Mirrors the speedups landed for Qwen3.5 0.8B / 2B (v1.8.0) and
// Qwen3-VL 2B Phase 1, applied to Granite's dense GQA decoder:
//   * MLState + slice_update KV cache per chunk (one buffer per body
//     chunk, written via ios18.slice_update inside the graph).
//   * mmap fp16 embed sidecar (Granite ships embedding_multiplier=12
//     pre-multiplied into `embed_weight.bin`, so Swift's lookup is a
//     plain memcpy — same kernel as Qwen3-VL stateful).
//   * Full-vocab fp16 logits out of `chunk_head` plus fp32 argmax
//     (with optional rep_penalty masking) in Swift — the workaround
//     that broke through the iPhone fp16 ANE reduction-bias ceiling
//     for Qwen3.5 v1.8.0. Granite's logits_scaling=10 is folded into
//     the lm_head conv weight at build time.
//
// Per-chunk inputs (matches conversion/build_granite4_chunks.py):
//   hidden_in   (1, 1, hidden) fp16
//   cos, sin    (1, 1, head_dim) fp16
//   causal_mask (1, 1, 1, max_seq) fp16   — -1e4 for slots > current_pos
//   current_pos (1,) int32
//   state       kv_cache_0 — managed by Core ML (makeState() per session)
//
// Head:
//   hidden_in (1, 1, hidden) fp16 → logits (1, 1, vocab) fp16
//
// Artifacts on disk (under model directory):
//   <dir>/granite4_decode_chunks/
//     embed_weight.bin
//     chunk_0.mlpackage / .mlmodelc
//     chunk_1.mlpackage / .mlmodelc
//     ... chunk_(num_chunks-1)
//     chunk_head.mlpackage / .mlmodelc
//   <dir>/model_config.json
//   <dir>/hf_model/   — tokenizer files (tokenizer.json, chat_template.jinja, ...)

import Accelerate
import CoreML
import Foundation
import Tokenizers


@Observable
public final class Granite4Generator: @unchecked Sendable {
    // MARK: - Config

    public struct Config: Sendable {
        public let maxSeq: Int
        public let vocab: Int
        public let hiddenSize: Int
        public let numLayers: Int
        public let numKVHeads: Int
        public let headDim: Int
        public let numBodyChunks: Int
        public let layersPerChunk: Int
        public let ropeTheta: Float
        public let bosTokenId: Int32
        public let eosTokenId: Int32
        public let computeUnits: MLComputeUnits

        /// Defaults matching ``ibm-granite/granite-4.1-3b`` →
        /// ``conversion/build_granite4_chunks.py`` output (5 chunks × 8 layers).
        public static let granite4_3b = Config(
            maxSeq: 2048, vocab: 100352,
            hiddenSize: 2560, numLayers: 40,
            numKVHeads: 8, headDim: 64,
            numBodyChunks: 5, layersPerChunk: 8,
            ropeTheta: 10_000_000,
            bosTokenId: 100257, eosTokenId: 100257,
            computeUnits: .cpuAndNeuralEngine)
    }

    // MARK: - Public observable state (UI-friendly).

    public var status: String = "Idle"
    public var running: Bool = false
    public var lastTokensPerSecond: Double = 0
    public var lastDecodeTokenCount: Int = 0
    /// Cumulative milliseconds spent inside each chunk's `prediction(...)`
    /// call across the last decode loop, plus head and embed/mask/RoPE
    /// fill. Indexed [chunk_0, chunk_1, …, chunk_{N-1}, head]. Reset at
    /// the start of every `stream()` call. Useful to spot which chunk is
    /// the bottleneck on iPhone vs Mac.
    public var lastPerChunkMs: [Double] = []
    public var lastHeadMs: Double = 0
    public var lastSamplerMs: Double = 0

    // MARK: - Private

    private var cfg: Config
    private let modelDir: URL
    private var tokenizer: (any Tokenizer)?

    private var bodyChunks: [MLModel] = []
    private var headChunk: MLModel?

    // mmap'd fp16 embed sidecar — vocab × hidden, multiplied by
    // embedding_multiplier at build time so the Swift lookup is a flat memcpy.
    private var embedMmapBase: UnsafeMutableRawPointer?
    private var embedMmapPtr: UnsafePointer<UInt16>?
    private var embedMmapLen: Int = 0
    private var embedMmapFD: Int32 = -1

    // Reusable single-step buffers + feature values.
    private var reusableHidden: MLMultiArray!
    private var reusableCos: MLMultiArray!
    private var reusableSin: MLMultiArray!
    private var reusableMask: MLMultiArray!
    private var reusablePos: MLMultiArray!
    private var fvHidden: MLFeatureValue!
    private var fvCos: MLFeatureValue!
    private var fvSin: MLFeatureValue!
    private var fvMask: MLFeatureValue!
    private var fvPos: MLFeatureValue!

    // Pre-allocated output buffers for outputBackings (iOS 16+). Avoids
    // Core ML allocating a fresh MLMultiArray per chunk per step (5
    // chunks × ~50 µs/alloc on A19 ≈ 0.25 ms / step, free win).
    // - chunkHiddenBuffers[i] receives chunk_i's "hidden" output.
    //   Each chunk writes its own buffer so we can keep a stable
    //   FeatureValue pointer to feed chunk_(i+1).
    // - headLogitsBuffer receives the head's "logits" output.
    private var chunkHiddenBuffers: [MLMultiArray] = []
    private var fvChunkHidden: [MLFeatureValue] = []
    private var headLogitsBuffer: MLMultiArray?
    private var fvHeadLogits: MLFeatureValue?
    // outputBackings dict expects raw MLMultiArray values (typed as Any
    // in the obj-c API). We set them via KVC to bypass the typed Swift
    // overload that wants [String: MLFeatureValue].
    private var bodyOutputBackings: [String: Any] = [:]
    private var headOutputBackings: [String: Any] = [:]

    // RoPE cos/sin tables (precomputed for 0..<2*maxSeq, half_dim).
    private var cosTable: [Float] = []
    private var sinTable: [Float] = []

    // MARK: - Init / load

    public init(modelDirectory: URL, cfg: Config = .granite4_3b) {
        self.modelDir = modelDirectory
        self.cfg = cfg
    }

    public func load(onProgress: ((String) -> Void)? = nil) async throws {
        status = "Loading tokenizer..."
        onProgress?(status)
        let tokDir = modelDir.appendingPathComponent("hf_model")
        if FileManager.default.fileExists(atPath: tokDir.path) {
            self.tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)
        }

        // Detect actual num_chunks from the bundle so the same Generator
        // works for 4/5/8/10-chunk variants (iPhone iOS 26.1 ANE compile
        // ceiling rejects 8-layer chunks; ship variant is 5 layers/chunk
        // = 8 chunks). model_config.json is the source of truth; fall
        // back to filesystem count if absent.
        let chunksDir = modelDir.appendingPathComponent("granite4_decode_chunks")
        let cfgURL = modelDir.appendingPathComponent("model_config.json")
        var detectedChunks = cfg.numBodyChunks
        if let data = try? Data(contentsOf: cfgURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let n = json["num_chunks"] as? Int {
            detectedChunks = n
        } else {
            // Fallback: count chunk_N files on disk.
            let fm = FileManager.default
            var n = 0
            while fm.fileExists(atPath: chunksDir
                .appendingPathComponent("chunk_\(n).mlmodelc").path)
                || fm.fileExists(atPath: chunksDir
                .appendingPathComponent("chunk_\(n).mlpackage").path)
            {
                n += 1
            }
            if n > 0 { detectedChunks = n }
        }
        if detectedChunks != cfg.numBodyChunks {
            print("[Granite4] adjusting numBodyChunks: cfg=\(cfg.numBodyChunks) → bundle=\(detectedChunks)")
            self.cfg = Config(
                maxSeq: cfg.maxSeq, vocab: cfg.vocab,
                hiddenSize: cfg.hiddenSize, numLayers: cfg.numLayers,
                numKVHeads: cfg.numKVHeads, headDim: cfg.headDim,
                numBodyChunks: detectedChunks,
                layersPerChunk: cfg.numLayers / detectedChunks,
                ropeTheta: cfg.ropeTheta,
                bosTokenId: cfg.bosTokenId, eosTokenId: cfg.eosTokenId,
                computeUnits: cfg.computeUnits)
        }

        status = "Resolving chunks..."
        onProgress?(status)
        guard let urls = resolveURLs() else {
            throw NSError(domain: "Granite4", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "granite4_decode_chunks/{embed_weight.bin, chunk_0..\(cfg.numBodyChunks - 1), chunk_head} missing under \(modelDir.path)"])
        }

        try mmapEmbed(urls.embed)

        let mcfg = MLModelConfiguration()
        mcfg.computeUnits = cfg.computeUnits

        bodyChunks.removeAll(keepingCapacity: true)
        for (i, u) in urls.body.enumerated() {
            status = "Loading chunk_\(i)..."
            onProgress?(status)
            let m = try MLModel(contentsOf: u, configuration: mcfg)
            bodyChunks.append(m)
        }
        status = "Loading chunk_head..."
        onProgress?(status)
        headChunk = try MLModel(contentsOf: urls.head, configuration: mcfg)

        try allocateBuffers()
        precomputeRopeTables()

        status = "Ready (\(bodyChunks.count) chunks + head, "
            + "ANE=\(cfg.computeUnits.rawValue))"
        onProgress?(status)
    }

    deinit {
        if let base = embedMmapBase, embedMmapLen > 0 {
            munmap(base, embedMmapLen)
        }
        if embedMmapFD >= 0 { close(embedMmapFD) }
    }

    /// Run one dummy step through every chunk + head on a throwaway
    /// MLState so the ANE compiles its dispatch cache before the first
    /// user generate(). On iPhone A19 the first call to each chunk pays
    /// ~1-3s of ANE compile; without prewarm that cost is averaged into
    /// the user's first chat tok/s reading and drags the apparent rate
    /// way below steady state. Mirrors Qwen3-VL Phase 1's `prewarm()`.
    public func prewarm() async throws {
        guard !bodyChunks.isEmpty, let head = headChunk else { return }
        memset(reusableHidden.dataPointer, 0, reusableHidden.count * 2)
        fillCosSin(forPosition: 0)
        fillCausalMask(forPosition: 0)
        setCurrentPos(0)

        let warmStates = bodyChunks.map { $0.makeState() }
        let opts = MLPredictionOptions()
        var hiddenFV: MLFeatureValue = fvHidden!
        for (ci, chunk) in bodyChunks.enumerated() {
            let prov = BodyProvider(
                hiddenIn: hiddenFV, cos: fvCos, sin: fvSin,
                mask: fvMask, pos: fvPos)
            let out = try await chunk.prediction(
                from: prov, using: warmStates[ci], options: opts)
            if let fv = out.featureValue(for: "hidden") { hiddenFV = fv }
        }
        let headProv = try MLDictionaryFeatureProvider(
            dictionary: ["hidden_in": hiddenFV])
        _ = try await head.prediction(from: headProv, options: opts)
    }

    // MARK: - Resolve / mmap

    private func resolveURLs() -> (body: [URL], head: URL, embed: URL)? {
        let fm = FileManager.default
        let dir = modelDir.appendingPathComponent("granite4_decode_chunks")
        let embed = dir.appendingPathComponent("embed_weight.bin")
        guard fm.fileExists(atPath: embed.path) else { return nil }

        func resolveOne(_ base: String) -> URL? {
            let mlc = dir.appendingPathComponent("\(base).mlmodelc")
            if fm.fileExists(atPath: mlc.path) { return mlc }
            let pkg = dir.appendingPathComponent("\(base).mlpackage")
            if fm.fileExists(atPath: pkg.path) {
                return try? MLModel.compileModel(at: pkg)
            }
            return nil
        }

        var body: [URL] = []
        for ci in 0..<cfg.numBodyChunks {
            guard let u = resolveOne("chunk_\(ci)") else { return nil }
            body.append(u)
        }
        guard let head = resolveOne("chunk_head") else { return nil }
        return (body, head, embed)
    }

    private func mmapEmbed(_ url: URL) throws {
        if let base = embedMmapBase, embedMmapLen > 0 {
            munmap(base, embedMmapLen)
            embedMmapBase = nil
        }
        if embedMmapFD >= 0 { close(embedMmapFD); embedMmapFD = -1 }

        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "Granite4", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "open embed: \(url.path)"])
        }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd)
            throw NSError(domain: "Granite4", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "fstat embed"])
        }
        let len = Int(st.st_size)
        guard let base = mmap(nil, len, PROT_READ, MAP_SHARED, fd, 0),
              base != MAP_FAILED else {
            close(fd)
            throw NSError(domain: "Granite4", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "mmap embed"])
        }
        embedMmapFD = fd
        embedMmapBase = base
        embedMmapLen = len
        embedMmapPtr = UnsafePointer(base.assumingMemoryBound(to: UInt16.self))
        let expected = cfg.vocab * cfg.hiddenSize * 2
        if len != expected {
            print("[Granite4] WARNING: embed size \(len) != expected \(expected) "
                + "(vocab=\(cfg.vocab) × hidden=\(cfg.hiddenSize) × 2)")
        }
    }

    // MARK: - Buffers

    private func allocateBuffers() throws {
        reusableHidden = try MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.hiddenSize)], dataType: .float16)
        reusableCos = try MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.headDim)], dataType: .float16)
        reusableSin = try MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.headDim)], dataType: .float16)
        reusableMask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: cfg.maxSeq)], dataType: .float16)
        reusablePos = try MLMultiArray(shape: [1], dataType: .int32)
        memset(reusableHidden.dataPointer, 0, reusableHidden.count * 2)
        memset(reusableCos.dataPointer, 0, reusableCos.count * 2)
        memset(reusableSin.dataPointer, 0, reusableSin.count * 2)
        memset(reusableMask.dataPointer, 0, reusableMask.count * 2)
        reusablePos.dataPointer.assumingMemoryBound(to: Int32.self)[0] = 0
        fvHidden = MLFeatureValue(multiArray: reusableHidden)
        fvCos = MLFeatureValue(multiArray: reusableCos)
        fvSin = MLFeatureValue(multiArray: reusableSin)
        fvMask = MLFeatureValue(multiArray: reusableMask)
        fvPos = MLFeatureValue(multiArray: reusablePos)

        // Pre-allocate one hidden output buffer per body chunk so
        // MLPredictionOptions.outputBackings can hand Core ML a stable
        // destination instead of allocating a fresh MLMultiArray per
        // chunk per step. Each chunk writes to chunkHiddenBuffers[i];
        // the next chunk reads from it as input. Apple's API forbids
        // aliasing the same buffer as both input and output of a
        // single prediction, hence one buffer per chunk (not a single
        // shared one).
        chunkHiddenBuffers.removeAll()
        fvChunkHidden.removeAll()
        for _ in 0..<cfg.numBodyChunks {
            let buf = try MLMultiArray(
                shape: [1, 1, NSNumber(value: cfg.hiddenSize)],
                dataType: .float16)
            memset(buf.dataPointer, 0, buf.count * 2)
            chunkHiddenBuffers.append(buf)
            fvChunkHidden.append(MLFeatureValue(multiArray: buf))
        }
        let logitsBuf = try MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.vocab)], dataType: .float16)
        memset(logitsBuf.dataPointer, 0, logitsBuf.count * 2)
        headLogitsBuffer = logitsBuf
        fvHeadLogits = MLFeatureValue(multiArray: logitsBuf)
        // Empty placeholders: we set the per-chunk binding inline in
        // runStep so each chunk gets its own buffer.
        bodyOutputBackings = [:]
        headOutputBackings = ["logits": logitsBuf]
    }

    private func precomputeRopeTables() {
        // RoPE cos/sin for positions [0, maxSeq) at the half-dim resolution.
        let halfDim = cfg.headDim / 2
        let theta = cfg.ropeTheta
        cosTable = Array(repeating: 0, count: cfg.maxSeq * halfDim)
        sinTable = Array(repeating: 0, count: cfg.maxSeq * halfDim)
        var invFreq = [Float](repeating: 0, count: halfDim)
        for i in 0..<halfDim {
            invFreq[i] = 1.0 / powf(theta, Float(2 * i) / Float(cfg.headDim))
        }
        for p in 0..<cfg.maxSeq {
            let pf = Float(p)
            for i in 0..<halfDim {
                let a = pf * invFreq[i]
                cosTable[p * halfDim + i] = cosf(a)
                sinTable[p * halfDim + i] = sinf(a)
            }
        }
    }

    // MARK: - Per-step setup

    private func embedLookup(token: Int32) {
        guard let ptr = embedMmapPtr else { return }
        let src = ptr + Int(token) * cfg.hiddenSize
        let dst = reusableHidden.dataPointer.assumingMemoryBound(to: UInt16.self)
        memcpy(dst, src, cfg.hiddenSize * 2)
    }

    private func fillCosSin(forPosition pos: Int) {
        let halfDim = cfg.headDim / 2
        let cBase = cosTable.withUnsafeBufferPointer { $0.baseAddress! + pos * halfDim }
        let sBase = sinTable.withUnsafeBufferPointer { $0.baseAddress! + pos * halfDim }
        let cPtr = reusableCos.dataPointer.assumingMemoryBound(to: UInt16.self)
        let sPtr = reusableSin.dataPointer.assumingMemoryBound(to: UInt16.self)
        // Lay out as the [first half, second half] mirror that
        // apply_rotary_pos_emb expects (rotate_half does
        // [x1, x2] → [-x2, x1]).
        for i in 0..<halfDim {
            let cf = cBase[i]
            let sf = sBase[i]
            let c16 = Float16(cf).bitPattern
            let s16 = Float16(sf).bitPattern
            cPtr[i] = c16
            cPtr[i + halfDim] = c16
            sPtr[i] = s16
            sPtr[i + halfDim] = s16
        }
    }

    private func fillCausalMask(forPosition pos: Int) {
        let mPtr = reusableMask.dataPointer.assumingMemoryBound(to: UInt16.self)
        let zero: UInt16 = 0
        // -1e4 in fp16 ≈ 0xF000 (sign + exp ≈ -10000 nearest representable).
        let neg = Float16(-10000).bitPattern
        for i in 0..<cfg.maxSeq {
            mPtr[i] = (i <= pos) ? zero : neg
        }
    }

    private func setCurrentPos(_ pos: Int) {
        reusablePos.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(pos)
    }

    // MARK: - Decode step

    private final class BodyProvider: NSObject, MLFeatureProvider {
        let fvHiddenIn: MLFeatureValue
        let fvCos: MLFeatureValue
        let fvSin: MLFeatureValue
        let fvMask: MLFeatureValue
        let fvPos: MLFeatureValue
        let featureNames: Set<String> = [
            "hidden_in", "cos", "sin", "causal_mask", "current_pos",
        ]
        init(hiddenIn: MLFeatureValue, cos: MLFeatureValue, sin: MLFeatureValue,
             mask: MLFeatureValue, pos: MLFeatureValue) {
            self.fvHiddenIn = hiddenIn
            self.fvCos = cos
            self.fvSin = sin
            self.fvMask = mask
            self.fvPos = pos
            super.init()
        }
        func featureValue(for n: String) -> MLFeatureValue? {
            switch n {
            case "hidden_in":    return fvHiddenIn
            case "cos":          return fvCos
            case "sin":          return fvSin
            case "causal_mask":  return fvMask
            case "current_pos":  return fvPos
            default: return nil
            }
        }
    }

    /// Push one token through chunks 0..N-1 + head. Caller manages
    /// `position`, `states`, and any sampling. Returns the head's
    /// fp16 logits over the full vocab. Caller does fp32 argmax.
    private func runStep(token: Int32, position: Int,
                         states: [MLState],
                         collectTimings: Bool) async throws -> MLMultiArray {
        guard let head = headChunk else {
            throw NSError(domain: "Granite4", code: 5,
                userInfo: [NSLocalizedDescriptionKey: "head not loaded"])
        }
        embedLookup(token: token)
        fillCosSin(forPosition: position)
        fillCausalMask(forPosition: position)
        setCurrentPos(position)

        // Per-chunk MLPredictionOptions with pre-allocated outputBackings
        // for "hidden". Avoids the per-step Core ML output MLMultiArray
        // allocation, which on A19 ANE is ~50 µs × 5 chunks = 0.25 ms /
        // step. iOS 16+ feature, lossless, free win.
        var hiddenFV: MLFeatureValue = fvHidden!
        for (ci, chunk) in bodyChunks.enumerated() {
            let t0 = collectTimings ? CFAbsoluteTimeGetCurrent() : 0
            let prov = BodyProvider(
                hiddenIn: hiddenFV, cos: fvCos, sin: fvSin,
                mask: fvMask, pos: fvPos)
            let opts = MLPredictionOptions()
            if ci < chunkHiddenBuffers.count {
                opts.setValue(["hidden": chunkHiddenBuffers[ci]],
                              forKey: "outputBackings")
            }
            let out = try await chunk.prediction(
                from: prov, using: states[ci], options: opts)
            // With outputBackings set, `out.featureValue(for: "hidden")`
            // returns the FeatureValue wrapping our pre-allocated buffer.
            // We fast-path to the cached fvChunkHidden[ci] when possible
            // to avoid creating a fresh MLFeatureValue wrapper.
            hiddenFV = (ci < fvChunkHidden.count)
                ? fvChunkHidden[ci]
                : (out.featureValue(for: "hidden") ?? hiddenFV)
            if collectTimings {
                let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                if ci < lastPerChunkMs.count {
                    lastPerChunkMs[ci] += dt
                }
            }
        }
        let tHead = collectTimings ? CFAbsoluteTimeGetCurrent() : 0
        let headProv = try MLDictionaryFeatureProvider(
            dictionary: ["hidden_in": hiddenFV])
        let headOpts = MLPredictionOptions()
        if let lb = headLogitsBuffer {
            headOpts.setValue(["logits": lb], forKey: "outputBackings")
        }
        let headOut = try await head.prediction(from: headProv, options: headOpts)
        let logits: MLMultiArray
        if let lb = headLogitsBuffer {
            logits = lb
        } else {
            guard let logitsFV = headOut.featureValue(for: "logits"),
                  let l = logitsFV.multiArrayValue else {
                throw NSError(domain: "Granite4", code: 7,
                    userInfo: [NSLocalizedDescriptionKey: "head no logits"])
            }
            logits = l
        }
        _ = headOut  // silence unused warning when fast-path hits
        if collectTimings {
            lastHeadMs += (CFAbsoluteTimeGetCurrent() - tHead) * 1000
        }
        return logits
    }

    // MARK: - Sampling (full-vocab fp32 argmax + rep_penalty)

    public struct SamplingOptions: Sendable {
        public var maxNewTokens: Int = 256
        public var temperature: Float = 0.0   // 0 → greedy
        public var topK: Int = 0              // 0 disabled
        public var repetitionPenalty: Float = 1.1
        /// Sliding window over GENERATED tokens for rep_penalty. Matches
        /// the Qwen3.5 v1.8.0 recipe (docs/QWEN35_FULL_VOCAB_REP_PENALTY.md):
        /// must NOT include the prompt — chat-template tokens otherwise
        /// get multi-count penalised and warp the vocab. 64 is the
        /// shipping value; raise if you see argmax loops, lower if
        /// natural repetition is being punished.
        public var repetitionWindow: Int = 64
        public var stopTokenIds: Set<Int32> = []
        public init() {}
    }

    /// fp16 logits → fp32 argmax with full-vocab rep_penalty masking.
    /// Mirrors `Qwen35MLKVGenerator.sampleFromFullLogits` per
    /// docs/QWEN35_FULL_VOCAB_REP_PENALTY.md. The rep_penalty must
    /// only target the **sliding window of generated tokens** (NOT
    /// the prompt) — penalising chat-template tokens repeatedly was
    /// the bug that warped Granite's Japanese decode on iPhone fp16
    /// ANE despite English staying clean.
    private func sample(from logits: MLMultiArray,
                        recent: [Int32],
                        options: SamplingOptions) -> Int32 {
        precondition(logits.count >= cfg.vocab, "logits.count < vocab")
        let src = logits.dataPointer.assumingMemoryBound(to: UInt16.self)

        // 1) fp16 → fp32 unpack via vImage (~1 ms for vocab=100K on iPhone).
        var fp32 = [Float](repeating: 0, count: cfg.vocab)
        let fp16Buf = UnsafeBufferPointer(start: src, count: cfg.vocab)
        var fp16Vec = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: fp16Buf.baseAddress!),
            height: 1, width: vImagePixelCount(cfg.vocab), rowBytes: cfg.vocab * 2)
        fp32.withUnsafeMutableBufferPointer { fbuf in
            var fp32Vec = vImage_Buffer(
                data: UnsafeMutableRawPointer(fbuf.baseAddress!),
                height: 1, width: vImagePixelCount(cfg.vocab), rowBytes: cfg.vocab * 4)
            _ = vImageConvert_Planar16FtoPlanarF(&fp16Vec, &fp32Vec, 0)
        }

        // 2) Full-vocab rep_penalty over the SLIDING WINDOW of recent
        //    generated tokens (NOT the prompt). HF-style: tokens with
        //    positive logits get divided down, negative ones get
        //    multiplied (more negative). Multi-counts are intentional —
        //    a token repeated within the window gets compounded penalty,
        //    which is how Qwen3.5 v1.8.0 escapes the fp16 ANE
        //    "おはる、おはる、…" loop on Japanese.
        if options.repetitionPenalty > 1.0 && !recent.isEmpty {
            let p = options.repetitionPenalty
            for tok in recent {
                let i = Int(tok)
                if i < 0 || i >= cfg.vocab { continue }
                let v = fp32[i]
                fp32[i] = (v > 0) ? v / p : v * p
            }
        }

        // 3) Greedy argmax via vDSP (or temperature-scaled topK sampling).
        if options.temperature <= 0 || options.topK == 1 {
            var bestV: Float = -.greatestFiniteMagnitude
            var bestIdx: vDSP_Length = 0
            fp32.withUnsafeBufferPointer { p in
                if let bp = p.baseAddress {
                    vDSP_maxvi(bp, 1, &bestV, &bestIdx, vDSP_Length(cfg.vocab))
                }
            }
            return Int32(bestIdx)
        }
        // Temperature + topK sampling — straightforward, only invoked
        // when caller opts in. Kept simple; no nucleus / typical / etc.
        let invT = 1.0 / max(options.temperature, 1e-4)
        for i in 0..<cfg.vocab { fp32[i] *= invT }
        let k = max(1, options.topK)
        let topK = topKIndices(fp32, k: k)
        let maxLogit = topK.map { fp32[$0] }.max() ?? 0
        var probs = [Float](repeating: 0, count: topK.count)
        var sum: Float = 0
        for (idx, i) in topK.enumerated() {
            let e = expf(fp32[i] - maxLogit); probs[idx] = e; sum += e
        }
        for i in 0..<probs.count { probs[i] /= sum }
        let r = Float.random(in: 0..<1)
        var cum: Float = 0
        for (idx, p) in probs.enumerated() {
            cum += p
            if r <= cum { return Int32(topK[idx]) }
        }
        return Int32(topK.last!)
    }

    private func topKIndices(_ x: [Float], k: Int) -> [Int] {
        let n = x.count
        guard k < n else { return Array(0..<n) }
        var idx = Array(0..<n)
        idx.sort { x[$0] > x[$1] }
        return Array(idx.prefix(k))
    }

    // MARK: - Public generate

    /// Stream tokens for a chat message list using the Granite chat
    /// template baked into the tokenizer (chat_template.jinja).
    public func stream(messages: [[String: String]],
                       options: SamplingOptions = SamplingOptions()
    ) async throws -> AsyncThrowingStream<String, Error> {
        // swift-transformers' Message is `[String: any Sendable]`. The
        // concrete `String` values flow in cleanly via Sendable.
        let castMessages: [Message] = messages.map { d in
            var out: Message = [:]
            for (k, v) in d { out[k] = v }
            return out
        }
        let inputIds = try applyChatTemplate(messages: castMessages)
        return try await stream(inputIds: inputIds, options: options)
    }

    public func stream(prompt: String,
                       options: SamplingOptions = SamplingOptions()
    ) async throws -> AsyncThrowingStream<String, Error> {
        return try await stream(
            messages: [["role": "user", "content": prompt]],
            options: options)
    }

    /// Lower-level entry point — caller hands in already-tokenized prompt.
    public func stream(inputIds: [Int32],
                       options: SamplingOptions = SamplingOptions()
    ) async throws -> AsyncThrowingStream<String, Error> {
        guard !bodyChunks.isEmpty, headChunk != nil else {
            throw NSError(domain: "Granite4", code: 8,
                userInfo: [NSLocalizedDescriptionKey: "load() not called"])
        }
        let cfg = self.cfg
        let tokenizer = self.tokenizer
        let stopIDs = options.stopTokenIds.union([cfg.eosTokenId])

        return AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else {
                    continuation.finish(throwing: NSError(
                        domain: "Granite4", code: 9,
                        userInfo: [NSLocalizedDescriptionKey: "deallocated"]))
                    return
                }
                do {
                    self.running = true
                    self.lastDecodeTokenCount = 0
                    self.lastTokensPerSecond = 0
                    self.lastPerChunkMs = Array(
                        repeating: 0, count: self.bodyChunks.count)
                    self.lastHeadMs = 0
                    self.lastSamplerMs = 0
                    defer { self.running = false }

                    // Fresh MLState per generation.
                    let states = self.bodyChunks.map { $0.makeState() }

                    // Serial T=1 prefill — mirrors the Qwen3.5 0.8B
                    // shipping path. Multifunction prefill_b<T> is a
                    // future commit; current 50 tok/s on M4 / 27.7 tok/s
                    // on iPhone 17 Pro is dominated by per-step decode,
                    // not prefill, for short prompts.
                    let promptLen = inputIds.count
                    var emitted: [Int32] = []
                    var recent: [Int32] = []
                    let recentWindow = max(1, options.repetitionWindow)
                    var nextToken: Int32 = 0
                    if promptLen > 0 {
                        for p in 0..<promptLen {
                            let tok = inputIds[p]
                            let logits = try await self.runStep(
                                token: tok, position: p, states: states,
                                collectTimings: false)
                            // The logits at position promptLen-1 predict
                            // the FIRST new token; we do a real sample
                            // there. Earlier positions ignore logits.
                            // recent[] is empty so no rep_penalty kicks in
                            // — first generated token is unbiased.
                            if p == promptLen - 1 {
                                nextToken = self.sample(
                                    from: logits, recent: recent,
                                    options: options)
                            }
                        }
                    }

                    // Decode loop. HF-TextStreamer pattern: re-decode the
                    // full emitted list each step, but **skip the emit**
                    // when the decoded text ends in U+FFFD (indicates
                    // incomplete byte sequence — the next token will
                    // complete it). Also handle the byte-level BPE quirk
                    // where swift-transformers' GPT2-style decoder may
                    // emit raw control chars in the U+0080–U+00FF range
                    // for byte tokens that haven't been re-mapped: skip
                    // the emit when the trailing run contains those.
                    // This eliminates the "Core��ML" garbling that
                    // simple per-token decoding produced.
                    func trailIsUnstable(_ s: String) -> Bool {
                        guard let last = s.unicodeScalars.last else { return false }
                        if last == "\u{FFFD}" { return true }
                        // Byte-level BPE leftover range (Latin Extended-A
                        // / control re-mapping): if the very last scalar
                        // sits here, more tokens may yet finish the
                        // grapheme. Only block on the trailing scalar to
                        // avoid hiding legitimate mid-string content
                        // (which is rare in English / Japanese).
                        let v = last.value
                        if v >= 0x0080 && v <= 0x00A0 { return true }
                        if v >= 0x0100 && v <= 0x0148 { return true }
                        return false
                    }
                    let decodeT0 = CFAbsoluteTimeGetCurrent()
                    var pos = promptLen
                    var emittedTextLen = 0
                    for _ in 0..<options.maxNewTokens {
                        if stopIDs.contains(nextToken) { break }
                        if pos >= cfg.maxSeq { break }
                        emitted.append(nextToken)
                        if let tk = tokenizer {
                            let allText = tk.decode(
                                tokens: emitted.map { Int($0) },
                                skipSpecialTokens: true)
                            // Hold the emit until the trailing chars are
                            // stable. Once the next token settles them,
                            // we emit everything since the last yield.
                            if !trailIsUnstable(allText)
                                && allText.count > emittedTextLen
                            {
                                let start = allText.index(
                                    allText.startIndex,
                                    offsetBy: emittedTextLen)
                                continuation.yield(String(allText[start...]))
                                emittedTextLen = allText.count
                            }
                        } else {
                            continuation.yield("[\(nextToken)]")
                        }
                        // Rolling tok/s update so the UI counter ticks up
                        // during the stream rather than only at EOS.
                        let dtSoFar = CFAbsoluteTimeGetCurrent() - decodeT0
                        if dtSoFar > 0 {
                            self.lastTokensPerSecond =
                                Double(emitted.count) / dtSoFar
                        }
                        self.lastDecodeTokenCount = emitted.count

                        // Append the just-emitted token to the rep_penalty
                        // window (nextToken hasn't been pushed yet — emitted
                        // is the most recent decoded token).
                        let lastEmitted = emitted.last ?? nextToken
                        recent.append(lastEmitted)
                        if recent.count > recentWindow {
                            recent.removeFirst(recent.count - recentWindow)
                        }
                        let logits = try await self.runStep(
                            token: nextToken, position: pos, states: states,
                            collectTimings: true)
                        let sT0 = CFAbsoluteTimeGetCurrent()
                        nextToken = self.sample(
                            from: logits, recent: recent,
                            options: options)
                        self.lastSamplerMs +=
                            (CFAbsoluteTimeGetCurrent() - sT0) * 1000
                        pos += 1
                    }
                    // Flush any held-back tail (e.g. last tokens left a
                    // trailing U+FFFD that now has no successor). Decode
                    // again and emit anything past `emittedTextLen`.
                    if let tk = tokenizer, !emitted.isEmpty {
                        let allText = tk.decode(
                            tokens: emitted.map { Int($0) },
                            skipSpecialTokens: true)
                        if allText.count > emittedTextLen {
                            let start = allText.index(
                                allText.startIndex,
                                offsetBy: emittedTextLen)
                            continuation.yield(String(allText[start...]))
                            emittedTextLen = allText.count
                        }
                    }

                    let dt = CFAbsoluteTimeGetCurrent() - decodeT0
                    self.lastDecodeTokenCount = emitted.count
                    if dt > 0 && emitted.count > 0 {
                        self.lastTokensPerSecond = Double(emitted.count) / dt
                    }
                    // Per-chunk + head + sampler timing summary so the
                    // user can read iPhone bottleneck info from the Xcode
                    // console without instruments. Reported as cumulative
                    // ms across the decode loop (not prefill) and average
                    // ms per token.
                    let n = max(1, emitted.count)
                    var report = "[Granite4] decode \(emitted.count) tokens "
                        + String(format: "in %.0f ms = %.1f tok/s\n", dt * 1000,
                                 self.lastTokensPerSecond)
                    for (ci, ms) in self.lastPerChunkMs.enumerated() {
                        report += String(format:
                            "  chunk_%d: %.0f ms total, %.1f ms/token\n",
                            ci, ms, ms / Double(n))
                    }
                    report += String(format:
                        "  head:    %.0f ms total, %.1f ms/token\n",
                        self.lastHeadMs, self.lastHeadMs / Double(n))
                    report += String(format:
                        "  sampler: %.0f ms total, %.1f ms/token",
                        self.lastSamplerMs, self.lastSamplerMs / Double(n))
                    print(report)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Apply Granite chat template via swift-transformers if the
    /// tokenizer was loaded.
    public func applyChatTemplate(messages: [Message]) throws -> [Int32] {
        guard let tk = tokenizer else {
            throw NSError(domain: "Granite4", code: 10,
                userInfo: [NSLocalizedDescriptionKey:
                    "tokenizer not loaded (hf_model/ missing)"])
        }
        let ids = try tk.applyChatTemplate(messages: messages)
        return ids.map { Int32($0) }
    }
}
