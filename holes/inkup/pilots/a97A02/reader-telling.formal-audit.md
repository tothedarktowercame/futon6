# a97A02 reader-oriented telling

## Formal audit

**[B0]** For \( 1 \le p < \infty \), \( h > 0 \) and
\( \mathrm{MemLp}\, f\, p \) on the restricted measure, the theorem
apm_a97a02 concludes the conjunction: \( \mathrm{ContinuousOn} \) of the
average apm_a97A02_average on \( [0,1] \), and the pointwise bound by
\( (2h)^{-1/p} \) times the \( \mathrm{eLpNorm} \) of \( f \). Both parts are
proved separately and joined by refine.

**[B1]** The measure is apm_a97A02_measure, volume restricted to
\( \mathrm{Icc}\ 0\ 1 \), and the average apm_a97A02_average divides the
\( \mathrm{Icc} \) integral by \( 2h \). The file's encoding note says that
this restriction automatically uses the zero extension outside \( [0,1] \):
integrating over the window against the restricted measure is integrating the
zero extension, so a window sticking past an endpoint contributes only its
part inside \( [0,1] \).

**[B2]** The lemma apm_a97A02_integrable_of_memLp derives
\( \mathrm{Integrable}\ f \) from \( \mathrm{MemLp} \) and \( 1 \le p \),
after installing the \( \mathrm{IsFiniteMeasure} \) instance for the
restricted measure by unfold and infer_instance.

**[B3]** The lemma apm_a97A02_measure_Icc_le bounds the restricted measure
of \( [t-h,t+h] \) by \( 2h \), by Measure.restrict_le_self followed by
Real.volume_Icc and linarith on the endpoints.

**[B4]** The lemma apm_a97A02_continuousOn_average installs the
\( \mathrm{NoAtoms} \) instance for the restricted measure and defines
\( F(b) = \int_{\mathrm{Iic}\ b} f \). The hypothesis \( \mathrm{hF} \) is
continuity of \( F \) on \( \mathrm{Iic}\ (1+h) \), from
IntegrableOn.continuousOn_Iic_primitive_Iic; this consumes integrability of
\( f \) (B2) and the NoAtoms instance, and the compositions
\( t \mapsto F(t\pm h) \) stay inside that half-line for
\( t \in [0,1] \). The file's own comment records why this route: writing an
\( \mathrm{Icc} \) integral as a difference of two \( \mathrm{Iic} \)
primitives avoids symmetric-difference bookkeeping.

**[B5]** The step \( \mathrm{hnum} \) rewrites the moving integral through
integral_Icc_eq_integral_Ioc and
intervalIntegral.integral_Iic_sub_Iic, making it
\( F(t+h) - F(t-h) \) on \( [0,1] \); hnum.div_const divides by \( 2h \).

**[B6]** The step \( \mathrm{hL1} \) applies
eLpNorm_le_eLpNorm_mul_rpow_measure_univ on the window-restricted measure,
then eLpNorm_mono_measure and the measure bound of B3, using the exponent
condition \( \mathrm{hexp} \) that \( 1 - 1/p \ge 0 \). The constant agrees
with the dual-norm reading: Hölder's factor \( (2h)^{1-1/p} \) divided by
\( 2h \) is \( (2h)^{-1/p} \).

**[B7]** The step \( \mathrm{hnormInt} \) uses
norm_integral_le_lintegral_norm, and \( \mathrm{hL1real} \) passes to reals
through ENNReal.toReal_mono, since neither factor is \( \top \). The final
calc divides by \( 2h \) and rewrites
\( 1 - 1/p + (-1) = -1/p \) by ring. The binder \( t \in [0,1] \) is
introduced by intro and never used in this branch: the estimate needs
measurability, the window bound and \( p \ge 1 \) only, so hL1real and the
final bound hold for every real \( t \).

**[B8]** The conjunction is assembled exactly as the two halves were proved:
the continuity component by apm_a97A02_continuousOn_average composed with
apm_a97A02_integrable_of_memLp, the bound component by the estimate chain,
for every \( h > 0 \) and \( 1 \le p < \infty \).
