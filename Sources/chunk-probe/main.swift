// ChunkProbe — single-chunk ANE-acceptance probe.
//
// Loads one compiled .mlmodelc with a chosen compute unit, builds zero-valued
// MLMultiArray inputs sized exactly to the model's input description, warms up,
// times predicts, and reports per-predict ms or the CoreML error (faithfully
// surfacing "error code: -1").
//
// Usage:
//   chunk-probe <model.mlmodelc> [ne|cpu|all|gpu] [iters]
//
// Calibrated by controls run through the same harness:
//   - chunk1.mlmodelc       ne  -> OK ~14ms   (known ANE-engaged in the smoke)
//   - chunk2_3way.mlmodelc   ne  -> ERROR -1  (known runtime rejector)

import CoreML
import Foundation

extension MLMultiArrayDataType {
    var footprint: Int {
        switch self {
        case .float16: return 2
        case .float32: return 4
        case .float64: return 8
        case .int32:   return 4
        default: return 4
        }
    }
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write(Data("usage: chunk-probe <model.mlmodelc> [ne|cpu|all|gpu] [iters]\n".utf8))
    exit(2)
}
let path = args[1]
let unitStr = args.count > 2 ? args[2] : "ne"
let iters = args.count > 3 ? (Int(args[3]) ?? 5) : 5

let units: MLComputeUnits
switch unitStr {
case "ne", "cpuane": units = .cpuAndNeuralEngine
case "cpu":         units = .cpuOnly
case "gpu":         units = .cpuAndGPU
case "all":         units = .all
default:
    FileHandle.standardError.write(Data("unknown unit \(unitStr)\n".utf8)); exit(2)
}

let url = URL(fileURLWithPath: path)
let cfg = MLModelConfiguration()
cfg.computeUnits = units

let model: MLModel
do {
    model = try MLModel(contentsOf: url, configuration: cfg)
} catch {
    print("[probe] LOAD_ERROR unit=\(unitStr) \(URL(fileURLWithPath: path).lastPathComponent): \(error)")
    exit(1)
}
print("[probe] loaded \(URL(fileURLWithPath: path).lastPathComponent) unit=\(unitStr)")

func makeInputs(_ desc: MLModelDescription) -> MLDictionaryFeatureProvider {
    var dict: [String: MLFeatureValue] = [:]
    for (name, input) in desc.inputDescriptionsByName {
        guard input.type == .multiArray, let mac = input.multiArrayConstraint else {
            FileHandle.standardError.write(Data("input \(name) not multiArray\n".utf8)); exit(2)
        }
        let arr = try! MLMultiArray(shape: mac.shape, dataType: mac.dataType)
        let bytes = arr.count * mac.dataType.footprint
        memset(arr.dataPointer, 0, bytes)
        dict[name] = MLFeatureValue(multiArray: arr)
    }
    return try! MLDictionaryFeatureProvider(dictionary: dict)
}

let provider = makeInputs(model.modelDescription)

// warmup (first call compiles for the chosen unit)
do { _ = try model.prediction(from: provider) } catch {
    print("[probe] WARMUP_ERROR unit=\(unitStr): \(error)")
    exit(1)
}
do { _ = try model.prediction(from: provider) } catch {
    print("[probe] WARMUP2_ERROR unit=\(unitStr): \(error)")
    exit(1)
}

let t0 = Date()
for _ in 0..<iters {
    _ = try model.prediction(from: provider)
}
let ms = Date().timeIntervalSince(t0) / Double(iters) * 1000
print(String(format: "[probe] OK unit=%@ %.2f ms/predict (%d iters)", unitStr, ms, iters))
exit(0)
