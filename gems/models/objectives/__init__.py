"""Pre-training repair channels composed into one spectrum-denoising objective.

ℒ = ℒ_mz (masked m/z) + λ_int·ℒ_int (masked intensity) + λ_rpd·ℒ_rpd (replaced-peak detection).
"""
