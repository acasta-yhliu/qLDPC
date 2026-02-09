from qldpc.circuits.alphasyndrome import AlphaSyndrome
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.codes.quantum import SurfaceCode

alphasyndrome = AlphaSyndrome(DepolarizingNoiseModel(0.005, include_idling_error=True), "pymatching")

circuit, _ = alphasyndrome.get_circuit(SurfaceCode(3, 3))

print(circuit)
