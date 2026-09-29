@_extern(c, "sin") func sine(_ x: Double) -> Double
@_extern(c, "cos") func cosine(_ x: Double) -> Double
@_extern(c, "exp") func exponential(_ x: Double) -> Double
@_extern(c, "log") func logarithm(_ x: Double) -> Double
@_extern(c, "tanh") func hyperbolic(_ x: Double) -> Double

func random(_ seed: inout UInt32) -> Double {
    seed = seed &* 1664525 &+ 1013904223
    return Double(seed) / 4294967296
}
