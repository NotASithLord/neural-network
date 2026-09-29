// Seven-layer dense network: two inputs, five tanh layers, one sigmoid output.
let weights = UnsafeMutablePointer<Float>.allocate(capacity: 264449)
let activations = UnsafeMutablePointer<Double>.allocate(capacity: 1281)
let deltas = UnsafeMutablePointer<Double>.allocate(capacity: 1281)
let display = UnsafeMutablePointer<Float>.allocate(capacity: 4096)
var neuralCount = 64, neuralSeed: UInt32 = 171
var neuralLoss = 0.0, neuralAccuracy = 0.0
var neuralSteps: Int32 = 0
func inputWidth(_ layer: Int) -> Int { layer == 0 ? 2 : neuralCount }
func outputWidth(_ layer: Int) -> Int { layer == 5 ? 1 : neuralCount }
func weightOffset(_ layer: Int) -> Int { layer == 0 ? 0 : neuralCount*3+(layer-1)*neuralCount*(neuralCount+1) }
@_cdecl("neural_init") public func neuralInit(_ count: Int32) {
    neuralCount = max(8,min(256,Int(count))); neuralSeed = 171; neuralSteps = 0
    for i in 0..<4096 { display[i] = 0 }
    for layer in 0..<6 {
        let ni = inputWidth(layer), no = outputWidth(layer), offset = weightOffset(layer)
        let scale = (6.0/Double(ni+no)).squareRoot()
        for j in 0..<no { for i in 0...ni {
            weights[offset+j*(ni+1)+i] = Float(i == ni ? (random(&neuralSeed)-0.5)*0.2 : (random(&neuralSeed)*2-1)*scale)
        }}
    }
}
func forward(_ x: Double, _ y: Double) -> Double {
    for layer in 0..<6 {
        let ni = inputWidth(layer), no = outputWidth(layer), offset = weightOffset(layer)
        for j in 0..<no {
            let base = offset+j*(ni+1)
            var sum = Double(weights[base+ni])
            for i in 0..<ni { sum += Double(weights[base+i])*(layer == 0 ? (i == 0 ? x : y) : activations[(layer-1)*neuralCount+i]) }
            activations[layer*neuralCount+j] = layer == 5 ? 1/(1+exponential(-max(-25,min(25,sum)))) : hyperbolic(sum)
        }
    }
    return activations[5*neuralCount]
}
@_cdecl("neural_step") public func neuralStep(_ count: Int32) {
    for _ in 0..<max(0,min(count,1024)) {
        let angle = random(&neuralSeed)*6.28318530718, r = random(&neuralSeed).squareRoot()*1.35
        let x = cosine(angle)*r, y = sine(angle)*r, target = r < 0.75 ? 1.0 : 0.0
        deltas[5*neuralCount] = forward(x,y)-target
        for layer in stride(from: 4, through: 0, by: -1) {
            for i in 0..<neuralCount {
                var sum = 0.0
                for j in 0..<outputWidth(layer+1) { sum += Double(weights[weightOffset(layer+1)+j*(neuralCount+1)+i])*deltas[(layer+1)*neuralCount+j] }
                let a = activations[layer*neuralCount+i]
                deltas[layer*neuralCount+i] = sum*(1-a*a)
            }
        }
        for layer in 0..<6 {
            let ni = inputWidth(layer), no = outputWidth(layer), offset = weightOffset(layer)
            for j in 0..<no { for i in 0...ni {
                let a = i == ni ? 1 : layer == 0 ? (i == 0 ? x : y) : activations[(layer-1)*neuralCount+i]
                weights[offset+j*(ni+1)+i] -= Float(0.025*deltas[layer*neuralCount+j]*a)
            }}
        }
        neuralSteps += 1
    }
}
@_cdecl("neural_evaluate") public func neuralEvaluate() {
    for cell in 0..<1024 { display[cell] = Float(forward(Double(cell%32)/31*2.7-1.35,Double(cell/32)/31*2.7-1.35)) }
    _ = forward(0.3,0.2)
    for i in 0..<(5*neuralCount) { display[1024+i] = Float(activations[i]) }
    neuralMeasure()
}
@_cdecl("neural_measure") public func neuralMeasure() {
    var loss = 0.0, correct = 0
    for i in 0..<1024 {
        let x = Double(i%32)/31*2.7-1.35, y = Double(i/32)/31*2.7-1.35
        let target = x*x+y*y < 0.5625 ? 1.0 : 0.0
        let p = max(0.0000001,min(0.9999999,Double(display[i])))
        loss -= target*logarithm(p)+(1-target)*logarithm(1-p)
        if (p >= 0.5) == (target == 1) { correct += 1 }
    }
    neuralLoss = loss/1024; neuralAccuracy = Double(correct)/1024
}
@_cdecl("neural_adapt") public func neuralAdapt(_ elapsed: Double,_ current: Int32,_ intensity: Int32) -> Int32 {
    let target = Double(max(1,min(3,intensity)))*4+2
    if elapsed < target*0.65 { return min(8,current+1) }
    if elapsed > target*1.3 { return max(1,Int32(Double(current)*0.7)) }
    return current
}
@_cdecl("neural_weights") public func neuralWeights() -> UnsafeMutablePointer<Float> { weights }
@_cdecl("neural_display") public func neuralDisplay() -> UnsafeMutablePointer<Float> { display }
@_cdecl("neural_loss") public func currentLoss() -> Double { neuralLoss }
@_cdecl("neural_accuracy") public func currentAccuracy() -> Double { neuralAccuracy }
