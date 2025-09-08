from fractions import Fraction

# Exact fractions
P_m3 = Fraction(1,15) + Fraction(1,20) + Fraction(1,25)  # P(m=3)
P_n5_m3 = Fraction(1,25)                                 # P(n=5 and m=3)
P_n5_given_m3 = P_n5_m3 / P_m3

print("P(n=5 | m=3) =", P_n5_given_m3, "≈", float(P_n5_given_m3))
print("P(m=3) =", P_m3, "≈", float(P_m3))
