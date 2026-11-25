# Week 4 gaps: statistics for the hybrid-vs-dense comparison

Date: 2026-11-25. Scope: fills the gaps left by sections 5 and 6 of `2026-11-25-week4-hybrid-retrieval.md` (not repeated here). English. Markers: `[computed]` = I computed it (simulation or arithmetic, assumptions stated in section 9); `[inference]` = my synthesis, not a source claim. Sources I could not read in full are listed under `## NEED DOWNLOAD` and are not relied on for numbers.

Simulation and extraction scripts were kept outside the repo (session scratch folder `.../scratchpad/stats-sim/`, not persistent). Everything needed to rebuild them is in section 9 and section 10.

## Summary

- **Voorhees & Buckley 2002 has no legal open copy** (ACM returned 403, NIST has the abstract only, OpenAlex and Semantic Scholar report closed). I read the open Buckley & Voorhees 2000 paper instead: at 50 topics and a 5% fuzziness value, Prec(1) had an error rate of 14.3% versus 3.6% for Prec(10) and 1.5% for AP, so top-1 measures are the least stable of all. Our R@1/RR endpoints are in that class.
- **McNemar:** Fagerland et al. recommend mid-p (two-sided mid-p = exact two-sided p minus the point probability of the observed count; equal discordant counts use 1 - f/2) and say not to use the exact conditional test "in any situation". [computed] For our structure (13 possible wins, 32 possible regressions) the unconditional size is at most 0.84% for exact and 2.0% for mid-p, so neither test is anti-conservative here; exact wastes power.
- **Permutation p-value convention:** with random sign flips, `(b+1)/(m+1)` is the valid p-value (Phipson & Smyth); it equals `B/w` once the identity assignment is counted (Hemerik & Goeman, Theorem 2). Enumerate all `2^k` sign patterns when the number of non-zero clusters `k <= 20`; otherwise use `m = 100,000` random flips.
- **Smucker 2007 and Urbano 2019 recommend the randomization/permutation test** (t-test as an alternative) and both include MRR/RR; neither studies clustered data or a binary R@1. Urbano finds RR "by far the most variable measure" and Type III errors up to 2% for P@10 and RR.
- **Query variations:** the only recommendation I could verify is UQV100's "double-averaging" (average variations within a topic, then across topics) for system performance; Culpepper et al. model variations as nested within topics. No source I read gives a validated significance recipe for exactly two phrasings per topic. Cluster-level testing is a defensible `[inference]`, consistent with Miller's clustered standard errors.
- **MRR status:** Fuhr calls RR ordinal; Sakai disagrees (known to me only through Ferrante et al. and Moffat; his paper is closed). Ferrante et al. show RR is the measure farthest from an interval scale and that interval-scale transforms change significance outcomes. Report R@1, MRR@10 and mean first rank (cap 11) together and say which one is confirmatory.
- **Tuning:** Hastie et al. section 7.10.2 (read via an Internet Archive copy) says any step that uses the labels must sit inside the cross-validation loop. Folds must be by cluster `[inference]`. Taylor et al. (CIKM 2006) found a 375-parameter ranker significantly worse than a 9-parameter one with 45 or fewer queries; one fusion parameter is the only affordable scale here.
- **Bruch et al.:** neither arXiv version quantifies a query count. Both say the convex-combination weight converges with "less than 5% of the training data"; the tuning-set sizes are sampled in 5% steps of the train split, so nothing finer is shown. The TOIS version could not be fetched.
- **Power, paraphrase R@1 (45 questions, 13 possible wins) [computed]:** to reach power 0.8 at alpha 0.05, the expected net gain must be about +6 (mid-p) or +7 (exact) with no regressions, +7.6 / +8.6 with one expected regression, +8.8 / +9.7 with two, and 4 or more expected regressions never reach 0.8.
- **Power, pooled MRR@10 with the cluster sign-flip test [computed]:** about +0.046 (no regressions) to +0.057 (one expected paraphrase regression plus 1.4 direct regressions) versus a hard ceiling of +0.0926. The pooled test is less powerful than the paraphrase-only test whenever regressions occur.
- **A sign-flip test whose non-zero clusters all point the same way is a sign test:** exact p = `2^(1-k)`, so at least 6 improved clusters are needed for p <= 0.05 [computed].
- **Plan-changing:** make the paraphrase stratum the confirmatory endpoint; do not decide on the bootstrap CI (it excluded 0 in 97.5% of datasets where the permutation test has power 0.80); pre-declare mid-p; with Holm over 2 contrasts the smallest detectable gains grow by about 0.007 MRR or 1 R@1 count (section 9).

---

## 1. Voorhees & Buckley, SIGIR 2002

**Access.** No legal full text found. `https://www.nist.gov/publications/effect-topic-set-size-retrieval-experiment-error` has the abstract and bib exports only (the page says to e-mail NIST for access problems). `https://dl.acm.org/doi/pdf/10.1145/564376.564432` returned HTTP 403 to an automated fetch. OpenAlex (`https://api.openalex.org/works/doi:10.1145/564376.564432`) and Semantic Scholar both report `oa_status: closed`. I did not use shadow libraries or ResearchGate. See `## NEED DOWNLOAD`.

**What I could read (abstract only).** Error rates are computed directly for topic sets up to 25 and extrapolated above that, from the number of topics and the observed difference in average scores; they are larger than anticipated. Source: `https://www.nist.gov/node/707561`.

**Closest open primary source: Buckley & Voorhees, "Evaluating Evaluation Measure Stability" (SIGIR 2000; reprinted SIGIR Forum 51(2), 2017).** `https://sigir.org/wp-content/uploads/2017/06/p235.pdf` (read, text extracted locally).

- Method (section 4): a "fuzziness" value is a relative difference; scores within, say, 5% of each other count as equal. For each pair of systems and each query set, the lesser of "A better" and "B better" counts is an error; error rate = errors / decisions, never above 50%.
- Table 2 (50 topics, 5% fuzziness): error rate in percent was 14.3 for Prec(1), 3.6 Prec(10), 2.9 Prec(30), 2.2 precision at 0.5 recall, 1.8 Prec(100), 1.5 AP, 1.3 R-Prec, 1.0 Prec(1000), 0.6 Recall(1000). Proportion of ties was 23.4% for Prec(1) and 12.8% for AP.
- Section 4.1: topic set sizes 5 to 50 were sampled; "For all measures the average error rate decreases as the number of topics increases", but the rate of decrease slows, "so that very large numbers of topics are required to force the error rate below a certain level". Prec(1) and Prec(2) are "more erratic".
- Section 4.2: larger fuzziness values lower the error rate at the cost of more ties.
- Section 5.1: the authors regard their rates as lower bounds (one collection, nine methods).

**Implications for about 45 clusters `[inference]`.**

- Our decisive metrics (R@1, RR) behave like Prec(1): the least stable class in their data. A difference smaller than the sampling noise of 45 clusters cannot be resolved; section 9 gives the detectable sizes.
- Their error rates concern system pairs on one TREC collection with relative differences; they do not transfer as numbers. The method does transfer: a cheap check on our own frozen set is the swap rate between the two halves of a random split of the 45 clusters, reported next to each comparison.
- I did not read the 2002 error-rate-versus-absolute-difference curves, so I state no numbers from them.

---

## 2. Fagerland, Lydersen & Laake 2013 (McNemar: mid-p versus exact conditional)

Source (read, open access): `https://pmc.ncbi.nlm.nih.gov/articles/PMC3716987/` (BMC Med Res Methodol 13:91).

- Notation in the paper: discordant counts `n12`, `n21` (here `b`, `c`), `n = b + c`; under H0, `b | n ~ Binomial(n, 1/2)`, point probability `f(x|n) = C(n,x)/2^n`.
- **Exact conditional, two-sided:** `p = 2 * sum_{x=0}^{min(b,c)} f(x|n)`; if `b = c`, `p = 1.0`.
- **Mid-p, two-sided (Eq. 5):** `p_mid = p - f(b|n)` (that is, double the one-sided mid-p `p1 - f/2`). If `b = c` (Eq. 6): `p_mid = 1 - f(b|n)/2`. I checked this against the paper's Table 6: for `b=1, c=7` exact 0.0703 and mid-p 0.0391; for `b=6, c=16` exact 0.0525 and mid-p 0.0347. My code reproduces all four [computed].
- **Recommendation (Discussion/Conclusions):** "We do not recommend use of the McNemar exact conditional test in any situation"; the mid-p test is "an excellent alternative to the vastly more complex exact unconditional test"; the asymptotic test is the most powerful but exceeds the nominal level in some scenarios. They advise against the continuity-corrected asymptotic test. The exact test's type I error rates were low, barely above 3% even at 100 pairs (paper's wording, paraphrased).
- **Type I error of mid-p:** "Even though the mid-p test does not guarantee preservation of the significance level, it did not violate the nominal level in any of the 9595 scenarios considered" (sample sizes N = 10, 15, ..., 100).
- Power (their N needed for 80% power at Delta = 0.25, averages): 42 asymptotic, 43 mid-p, 43 exact unconditional, 49 exact conditional. They state these averages are not for planning a study.

**Two caveats for this project [computed].**

1. Conditional on a fixed number `n` of discordant pairs, the mid-p test can exceed 5%: e.g. 6.25% at `n=5`, 7.03% at `n=8`, 7.68% at `n=16`. The nominal-level claim is for the unconditional (mixed over `n`) distribution.
2. For our design (baseline paraphrase R@1 is 32 hits and 13 misses, so `b <= 13`, `c <= 32`), the unconditional size, under independent per-question conversions with E[b] = E[c], is at most 0.84% for the exact test and 2.04% for mid-p over the whole null range (e.g. 0.82% / 2.04% at E[n] about 7.8). Neither test is anti-conservative here; the exact test just wastes power. This is a model-based size, not a proof for the real, dependent data.

**Recommendation `[inference]`:** pre-declare mid-p as the decision statistic and report the exact p next to it as the conservative bound. Never use the asymptotic chi-square form (fewer than 20 discordant pairs expected).

---

## 3. Phipson & Smyth 2010 ("Permutation p-values should never be zero")

Source (read in full, arXiv preprint of the SAGMB article): `https://arxiv.org/pdf/1603.05766`.

- **The rule.** Treat the permutation test as generating an exact discrete null distribution. If `m` random permutations are drawn without repetition and `b` is the number with a statistic at least as extreme as observed, the exact p-value is `p_u = (b+1)/(m+1)` (Section 5). The naive estimate `B/m` has actual type I error `(floor(m*alpha)+1)/(m+1)` (Section 2), which exceeds alpha for small alpha, and it can be exactly zero, which "makes no inferential sense".
- **Exhaustive enumeration:** the exact p-value is `(b_t+1)/(m_t+1)` with `m_t` the number of distinct values not counting the original (Section 5). With sampling **with replacement** the exact p-value `p_e` is slightly smaller than `(b+1)/(m+1)`; `p_u` "can be very conservative" when `m_t` is small (Fig. 1a, `m_t = 126`) and over-estimates a 0.05 p-value by about 2% at `m_t = 1000`.
- **Recommendations (Section 7):** enumerate or sample without replacement when feasible (it has more power than sampling with replacement); never use the unbiased `B/m` as a p-value; use `p_u` at minimum, or `p_e` when `m` is not negligible relative to `m_t`.
- **Identity must be counted.** Hemerik & Goeman, "Exact testing with random permutations" (`https://arxiv.org/pdf/1411.7565`, read in part): for a group of transformations, random draws that include the identity give a test of level at most alpha (Theorem 2), and "the formula (B+1)/(w+1) simplifies to B/w if the identity map is added to the random permutations". Sign flips of cluster differences form such a group (`2^k` elements), so `(b+1)/(m+1)` with `b` counting random flips (not the observed pattern) is the same as `B/w` with the observed pattern counted among `w = m+1`.
- **When enumeration is preferable `[inference]`:** when `2^k` is at most about 1 million (`k <= 20`) enumeration is cheap, exact and removes Monte Carlo error. It also avoids the conservative `(b+1)/(m+1)` bias that matters when the null has few distinct values (all-positive differences give only two extreme patterns). Above `k = 20`, use `m = 100,000` random flips: the Monte Carlo standard error at p = 0.05 is about 0.0007, and the smallest reportable p is 1e-5.
- Smucker et al. computed their sampled p-values as the plain fraction (`(689+691)/100000 = 0.0138`, no +1); Urbano et al. used 1,000,000 replicas. Neither is wrong at those sizes, but our code should follow the `+1` rule. Sources: `https://ciir-publications.cs.umass.edu/getpdf.php?id=744` and `https://arxiv.org/pdf/1905.11096v2`.

---

## 4. Smucker, Allan & Carterette (CIKM 2007) and Urbano, Lima & Hanjalic (SIGIR 2019)

Both read in full. Earlier notes covered the headline; here is what the questions add.

- **Smucker et al. recommendation (confirmed):** "we recommend the randomization test be used when it is applicable. The randomization test is applicable to most IR experiments." Wilcoxon and sign "should be discontinued for measuring the significance of a difference between means". Source: `https://ciir-publications.cs.umass.edu/getpdf.php?id=744`.
- **Does it cover per-query reciprocal rank? Yes, in one paragraph.** Section 5.3: "We also looked at how the precision at 10 (P10), mean reciprocal rank (MRR), and R-precision affected the results. In general the tests behaved the same as for the MAP. Of note, the Wilcoxon test showed less variation for the MRR than for the other metrics." Same source. No detail or tables for MRR are given.
- **Urbano et al. recommendation (confirmed):** the t-test and the permutation test "maintain the Type I error rate at the alpha level across measures, topic set sizes and significance levels"; the t-test is the top recommendation for simplicity but the permutation test "can accommodate other test statistics besides the mean"; bootstrap-shift is biased toward small p-values; Wilcoxon and sign "should be discontinued". Source: `https://arxiv.org/pdf/1905.11096v2`.
- **RR is covered:** measures are AP, nDCG@20, ERR@20, P@10 and RR; topic set sizes 25, 50, 100. "For RR we see very similar results ... The permutation test shows nearly ideal behavior for RR too" (Section 4.1). At 50 topics and alpha 0.05 the permutation and t-tests keep Type I error at 0.05, bootstrap-shift 0.059 (two-tailed). "RR is by far the most variable measure and hence more likely to observe outliers that lead to wrong conclusions" (Section 4.3); Type III errors "for measures like P@10 and RR, or small topic sets ... can be as high as 2%". Same source.
- **What neither covers:** correlated or clustered topics (their topics are independent), a binary R@1, or a measure with only a few distinct values like ours (R@1 differences take values in {-1, -0.5, 0, 0.5, 1}). `[inference]` Their sign-test warning is about continuous measures with ties dropped; for a binary per-question outcome the sign test on discordant pairs is simply McNemar, and no source says to discontinue it.
- **Design consequence `[inference]`:** the cluster-level permutation test is the Smucker/Urbano test with clusters as topics, so their Type I findings carry over only to the extent that cluster differences are independent and exchangeable around zero under H0.

---

## 5. Significance testing with query variations

### 5.1 What the primary sources say

- **UQV100 (Bailey, Moffat, Scholer, Thomas, SIGIR 2016)** (read, `https://static.aminer.org/pdf/20170130/pdfs/sigir/vmcnjydfiexh39qap6pl8drb0jryleo2.pdf`): "When assessing the latter [per query or per backstory basis], we recommend averaging the query-level performance across all query variations belonging to a specific backstory first, and then averaging these across the 100 backstories (double-averaging)." This is a recommendation on aggregation. The paper does not give a significance test.
- **Culpepper, Faggioli, Ferro, Kurland, "Topic Difficulty: Collection and Query Formulation Effects"** (TOIS; author-hosted preprint read, `https://www.dei.unipd.it/~ferro/papers/2021/TOIS2021-CFFK.pdf`): ANOVA with query formulations "as a nested factor within topics", because "every query formulation for this topic is not independent, it cannot be treated as a separate factor". They note that Bailey et al. had treated variations and topics as separate factors, "which may be at odds with independence assumptions made by the standard ANOVA model". Query formulation has an effect size comparable to the topic factor and larger than the system factor in their data.
- **Moffat, Scholer, Thomas, Bailey, CIKM 2015, "Pooled evaluation over query variations"** (read, `https://minerva-access.unimelb.edu.au/server/api/core/bitstreams/47449c02-3623-5315-8240-9556b98318ca/content`): about pooling; the effect of query variation on pool size is as strong as the effect of system variation. Not a significance-testing source.
- **Bailey, Moffat, Scholer, Thomas, SIGIR 2015, "User variability and IR system evaluation"** (abstract only, `https://people.eng.unimelb.edu.au/ammoffat/abstracts/bmst15sigir.html`): the score range of one system across query variations is comparable to or larger than the range across systems with a single query per topic. Full text not read.
- **Benham, Mackenzie, Moffat, Culpepper, "Boosting Search Performance Using Query Variations"** (read, `https://www.arxiv.org/pdf/1811.06147`): significance is "computed using the Bonferroni-corrected paired t-test"; the text I read does not say whether the unit is the query or the topic, so I draw no conclusion. They note a topic-effect bias in single-query splits.
- **Zuccon, Palotti, Hanbury, CIKM 2016** (not read; mean-variance framework known only from secondary descriptions) and **Faggioli, Ferro et al.** beyond the TOIS paper: see `## NEED DOWNLOAD`.
- **Cluster-robust standard errors:** Miller, "Adding Error Bars to Evals" (read, `https://arxiv.org/pdf/2411.00640`): Eq. 8 gives `SE_clustered = (1/n) * sqrt(sum_c (sum_i (d_ic - dbar))^2)`, computed from paired differences; clustered SEs "can be over 3X larger than naive standard errors" (Table 4, DROP). Miller says bootstrapping is unnecessary unless the scheme is complicated.
- **Cluster bootstrap/permutation theory:** Field & Welsh 2007 (JRSS-B) is behind the publisher paywall and ANU lists the file as "request a copy", restricted until 2037. Abstract (via search): the cluster bootstrap is consistent under both the transformation and random-effect models, whereas the residual bootstrap is consistent only under the transformation model. Davison & Hinkley 1997 is a book I could not open. Both are in `## NEED DOWNLOAD`.

### 5.2 Is "average per topic, then test over topics" the accepted practice?

- **Accepted for aggregation:** yes, as a recommendation by UQV100's authors (double-averaging). **Accepted for significance testing:** I found no primary source that states it; the nearest are the nested ANOVA model and cluster-robust SE logic.
- `[inference]` With exactly two questions per cluster, equal sizes, and a paired design, testing the 45 cluster means is equivalent in point estimate to the per-question mean difference and is the simplest valid cluster analysis: it respects the dependence between phrasings of one gold answer, needs no variance-component model, and matches Miller's clustered SE up to a degrees-of-freedom factor (Miller's Eq. 8 for equal clusters of 2 is `sqrt(sum_c (m_c - dbar)^2) / C`, with `m_c` the cluster means, versus the sample SE `sqrt(sum_c (m_c - dbar)^2 / (C-1) / C)`).
- The price is that the two phrasings are collapsed: direct (easy, lexical overlap) and paraphrase (hard) are different strata, and a pooled cluster mean dilutes the informative one (section 9). Prefer a stratum-level test (45 clusters of one question) for the confirmatory claim, and keep pooled results descriptive.

---

## 6. Sakai's reply to Fuhr and Ferrante, Ferro & Fuhr: status of MRR

**Sakai (SIGIR Forum 54(1), 2020, "On Fuhr's guideline for IR evaluation")**: OpenAlex reports `oa_status: closed` (`https://api.openalex.org/works/doi:10.1145/3451964.3451976`), but the SIGIR Forum June 2020 issue page (`https://sigir.org/forum/issues/june-2020/`) links a free copy, `http://sigir.org/wp-content/uploads/2020/06/p14.pdf`. *Update 2026-11-25: the coordinator read it in full (8 pages).* What it says:

- §2.1 "Thou Shalt Not Compute MRR nor ERR?": "I for one disagree." On means of ordinal scales: "by no means a universal truth". On RR: "it is also not clear to me whether RR really cannot be considered as an interval-scale measure"; RR "*considers*" equal rank deltas equal "according to its user model"; "I do not see why researchers should be *forbidden* to use RR and ERR."
- §2.2 (MAP): AP should not be "banned altogether", though graded measures are preferable when graded judgements exist.
- §2.4 (relative improvements): "I do not know if this should be forbidden. But I prefer reporting effect sizes, more specifically, standardised mean differences such as Hedge's g and Glass's Δ".
- §2.5 (simple holdout): agrees "to the extent that researchers should be aware that different training/validation/test splits may yield different results".
- §2.3, §2.6 (no hypotheses after the experiment), §2.7 (correct for multiple tests), §2.8 (report effect sizes), §2.9, §2.10: "Agreed."
- Implication `[inference]`: MRR@10 can stay as a secondary column beside R@1; both authors agree on one pre-declared hypothesis, correction when several are tested, and effect sizes, which is what the Week 4 pre-registration does.

Second-hand descriptions read earlier (consistent with the paper):

- Ferrante, Ferro, Fuhr: "Fuhr took a firm position and argued that Mean Reciprocal Rank (MRR) should not be computed because: 1. in general, RR is just an ordinal scale ... 2. in particular, RR has some counter-intuitive behaviour. On the other hand, Sakai has recently disagreed with Fuhr: 1. in general, on the fact that means should not be computed for an ordinal scale ... 2. in particular, on the use of RR which Sakai finds quite useful from a practical point of view." Source: `https://arxiv.org/pdf/2101.02668` (discussion of Mean Reciprocal Rank).
- Moffat quotes Sakai: "it is also not clear to me whether RR really cannot be considered as an interval-scale measure", and that Sakai "has also voiced opinions in support of Fuhr's comments in regard to statistical testing". Source: `https://arxiv.org/pdf/2207.03103` (Sections 4.1 and 4.4).

**Ferrante, Ferro & Fuhr, "Towards meaningful statements in IR evaluation" (IEEE Access 2021; read: `https://arxiv.org/pdf/2101.02668`):**

- Example 9: system A with first relevant at ranks 1 and 4 has MRR 0.625 and system B at rank 2 twice has MRR 0.5, but mean first rank is 2.5 for A and 2 for B; the conclusions reverse.
- Among measures not depending on the recall base, RR is the farthest from an interval scale (Kendall tau between RR and its interval-scale transform 0.67 to 0.93), "a drop in 'intervalness' in the range 7% to 33%".
- Interval-scale transforms change significance outcomes; for RR the shift in significant/non-significant run pairs (their delta%) is "up to a 72%" and "over 90%" of runs sit on just 4 possible RR values (their significance-test analysis and Figure 3).
- Their recommended practice, neutral in stance: check whether conclusions change under the transformation and report that sensitivity; do not claim a particular scale is "right".

**Moffat's rebuttal (`https://arxiv.org/pdf/2207.03103`):** RR is an interval scale if users perceive rank-1, rank-2 and no-hit as equally spaced in usefulness; an argument against RR "must be justified based on rhetoric (or on data) about user perceptions of usefulness". He also says a metric must be chosen first and a test second.

**What to report with MRR `[inference]`:**

1. R@1 and R@5 counts per stratum, with paired discordant counts (b, c).
2. MRR@10 and mean first rank (cap 11 for a miss, declared in advance), with the same test on both. If the two disagree, say so (the H1 vector in section 10 shows R@1 and MRR giving different verdicts on the same data).
3. The rank distribution (histogram), which is what the eval report already prints.
4. State the unit of analysis (cluster) and the number of clusters, not only questions.

---

## 7. Hastie, Tibshirani & Friedman, section 7.10.2; IR tuning with few queries

**Access note.** The authors' host (`https://hastie.su.domains/Papers/ESLII.pdf`) refused connections from this machine. I read the text from the Internet Archive capture of that same URL (`https://web.archive.org/web/2023id_/https://hastie.su.domains/Papers/ESLII.pdf`), PDF pages 264 to 266 (book pages 245 to 247). Page numbers refer to that capture.

- **Wrong way** (p. 245): screen predictors on all samples, build a classifier, then cross-validate. With N = 50 samples, p = 5000 independent Gaussian predictors, the 100 best predictors by correlation and a 1-nearest-neighbour classifier, "the average CV error rate was 3%. This is far lower than the true error rate of 50%." Reason: the predictors "have already seen" the left-out samples.
- **Right way** (pp. 245-246): split into K folds; for each fold, do the screening and the fitting using only the other folds, then predict the held-out fold.
- **Principle** (p. 246): "with a multistep modeling procedure, cross-validation must be applied to the entire sequence of modeling steps. In particular, samples must be 'left out' before any selection or filtering steps are applied." Unsupervised screening that does not use labels may be done before. Quote: "we have seen this blunder committed many times in published papers in top rank journals."

**Mapping to this project `[inference]`:** choosing alpha, a tokenization, a threshold or a failure-driven regex fix by looking at the frozen 90-question results and then reporting the same set is the wrong way. Right way: fix everything a priori (as section 6 of the earlier notes says) or run a pre-declared K-fold CV in which the whole choice happens inside each training fold. Folds must be by cluster, so the direct and paraphrase phrasings of one gold answer never straddle a train/test boundary (otherwise paraphrase test questions leak their direct twin). With 45 clusters and 5 folds each held-out fold has 9 clusters, so the CV estimate carries a standard error of the same size as the effect (section 9).

**IR-specific guidance on few queries (found):**

- Taylor, Zaragoza, Craswell, Robertson, Burges, "Optimisation methods for ranking functions with multiple parameters" (CIKM 2006; read, `https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/optimisation_multiple_parameters.pdf`): a common TREC situation is 50 training queries, and "the question of whether one can reasonably train on less than (say) 100 queries may be critical". Training sets of 16 to 2048 queries were used. "Paired t-tests indicate that the 375-parameter model significantly outperforms the 9-parameter model for training sets of 256 queries or more. It is also significantly worse for training sets of 45 queries or fewer."
- Fuhr 2017 (already in the earlier notes): tuning set disjoint from the test set; complete hypothesis list declared in advance.
- No source gave a rule such as "n queries per tuned parameter". Conclusion: tune at most one scalar, and only with a pre-declared CV; otherwise fix it a priori.

---

## 8. Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval"

Versions: arXiv v1 (21 Oct 2022) and v2 (4 May 2023) (`https://arxiv.org/abs/2210.11934`); ACM TOIS version DOI `10.1145/3596512`, listed by OpenAlex as bronze open access (`https://api.openalex.org/works/doi:10.1145/3596512`) but the ACM PDF returned HTTP 403 to automated fetches (`https://dl.acm.org/doi/pdf/10.1145/3596512`), so I did not read the TOIS text.

- **Both arXiv versions, read in full (`https://arxiv.org/pdf/2210.11934v1` and `https://arxiv.org/pdf/2210.11934`):** the abstract says convex combination "is sample efficient, requiring only a small set of training examples to tune its only parameter"; the Introduction says "just a handful of labeled queries". The only quantification is in the section "Interpretability and Sample Efficiency" (with Fig. 12): "with less than 5% of the training data, which is often a small set of queries, TM2C2's alpha converges, regardless of the magnitude of domain shift". The experiment samples progressively larger subsets of the train split "with a step size of 5%", tunes, and evaluates NDCG@1000 on the test split, averaged over 5 trials with 95% confidence intervals.
- **So no version gives a number of queries.** Tuning sets are sampled in 5% steps of each dataset's train split, so nothing finer than 5% is tested; the paper's Table 1 lists test query counts (e.g. MS MARCO dev 6,980; NFCorpus 323; SciFact 300) but I did not find train-split sizes in the text, so I cannot turn "5%" into a count. The objective was NDCG@1000, not R@1 or MRR.
- **Implication `[inference]`:** the claim "a handful of queries" cannot be used to justify tuning alpha on 45 clusters. The result says the curve is flat by the time the 5% grid reaches it; it does not say how few queries suffice. Our 45 clusters are below any evidence given. Use the alpha value fixed a priori from the literature (the earlier notes: 0.7 or 0.8, the Bruch paper itself uses 0.8 as the reference in its Fig. 12) and run CV only as a labeled sensitivity check.

---

## 9. Simulation: smallest detectable improvements

### 9.1 Assumptions (all of them)

- Baseline ranks (from the task): direct = 44 at rank 1, 1 at rank 2; paraphrase = 32 at rank 1, 6 at rank 2, 5 at rank 3, 2 at rank 4; no misses. Check: pooled MRR@10 = (44.5 + 37.1667)/90 = 0.90741 [computed], matching 0.907.
- Cluster = one direct plus one paraphrase question. The true pairing is not known to me; I used pairing A (the single direct rank-2 cluster has paraphrase rank 1) and pairing B (it has paraphrase rank 2). Results differ by 0.005 MRR or less, except where noted.
- **Hybrid model.** Each baseline question with rank greater than 1 independently becomes rank 1 with probability `q_up` ("win", the largest possible RR gain per fixed question; partial moves, e.g. rank 4 to 2, give less MRR gain for the same R@1 count, so the MRR thresholds below are lower bounds); otherwise it stays. Each baseline rank-1 question (direct and paraphrase) independently drops to rank 2 with probability `q_down` ("regression"). Variant "shared": one drop draw per cluster applied to both its rank-1 questions.
- `q_down` = 0, 1/32, 2/32, which gives expected paraphrase regressions 0, 1, 2 and expected direct regressions 0, 1.4, 2.75.
- **Paraphrase R@1 McNemar (a):** `b ~ Binomial(13, p_w)` wins, `c ~ Binomial(32, p_c)` regressions, exact enumeration over (b, c). Reject when two-sided p <= alpha and `b > c` (the wrong-direction rejection is negligible). `E[c]` = 32 `p_c`; expected net gain = `13 p_w - E[c]`. Reported value = smallest `E[b]` such that power is at least 0.8 for every larger `p_w` (monotone envelope).
- **Pooled MRR@10 (b):** per-cluster `d_c` = mean of RR differences of the two questions; statistic = sum of `d_c`; two-sided p from 10,000 random sign flips with `(b+1)/(M+1)` (one fixed sign matrix, `default_rng(20261125)`, reused for all datasets; each p-value is individually valid); 2000 simulated datasets per scenario (power standard error about 0.009); `q_up` grid 0 to 1 step 0.05; the reported threshold is the linear-interpolated crossing of 0.8 after which power stays above 0.8. The gain is the analytic expected `Delta`, not the sample value.
- Ceiling: all 13 paraphrase misses and the direct rank-2 question fixed gives pooled MRR 1.0, a gain of 0.0926 (paraphrase stratum +0.1741) [computed].
- Not modelled: win probabilities that vary by question; correlation between phrasing outcomes in wins (direct questions are almost all already rank 1, so little room); article_ref and out-of-scope strata; a different baseline-pairing structure than A/B.

### 9.2 (a) Paraphrase R@1, McNemar, alpha = 0.05 [computed]

Discrete frontier: smallest `b` with p <= 0.05 and `b > c` (cap `b <= 13`):

| regressions c | exact: min b | mid-p: min b |
|---|---|---|
| 0 | 6 | 5 |
| 1 | 8 | 7 |
| 2 | 10 | 9 |
| 3 | 12 | 11 |
| 4 | 13 | 12 |
| 5 or more | impossible | impossible |

Power 0.8 (binomial model; expected values):

| E[c] | exact: E[b], net E[b-c] | mid-p: E[b], net E[b-c] |
|---|---|---|
| 0 | 7.0, +7.0 | 6.0, +6.0 |
| 1 | 9.6, +8.6 | 8.6, +7.6 |
| 2 | 11.7, +9.7 | 10.8, +8.8 |
| 3 | 12.9, +9.9 | 12.4, +9.4 |
| 4 | max power 0.629 | max power 0.629 |
| 5 | max power 0.425 | max power 0.425 |

With Holm over 2 contrasts the first step is alpha = 0.025 (Monte Carlo, `E[c]` 0 and 1): mid-p needs net about +7.0 and +8.6; exact needs about +8.0 and +9.5.

The direct stratum (44 of 45 at rank 1) can show at most one win: an improvement is untestable there, and only regressions can be detected (c = 5 with b = 0 gives exact 0.0625, mid-p 0.0312).

### 9.3 (b) Pooled MRR@10, cluster sign-flip, alpha = 0.05 [computed]

Smallest expected pooled improvement with power at least 0.8 (pairing A; in parentheses, the paraphrase-stratum MRR gain and expected paraphrase R@1 net gain at that point):

| scenario | pooled MRR (cluster sign-flip) | paraphrase-only MRR (sign-flip, 45 clusters) | pooled R@1 (cluster sign-flip) |
|---|---|---|---|
| no regressions | +0.046 (+0.087; +6.5) | +0.050 (+0.094; +7.0) | +0.046 |
| 1 para + 1.4 direct regressions, independent | +0.057 (+0.121; +8.8) | +0.047 (+0.102; +7.4) | +0.068 |
| same, shared per cluster | +0.063 (+0.133; +9.7) | +0.046 (+0.100; +7.3) | +0.079 |
| 2 para + 2.75 direct regressions, independent | +0.063 (+0.146; +10.6) | +0.045 (+0.111; +7.9) | not reached before the ceiling |
| same, shared per cluster | not reached (power 0.67 at the ceiling) | +0.043 (+0.108; +7.8) | not reached |

Pairing B: no regressions +0.048; 1 regression +0.058 (independent) and +0.067 (shared); 2 regressions: power 0.797 only at the ceiling.

Holm first-step alpha = 0.025 (pairing A, `E[c]` 0 and 1, independent): pooled MRR +0.054 and +0.064; paraphrase-only MRR +0.057 and +0.055.

Read-outs:

- When all non-zero clusters improve, the test is a sign test: all-positive differences yield exact p = `2^(1-k)`, so power 0.8 is reached when about 6 clusters improve, for any size of gain. In that regime the pooled MRR test, the pooled R@1 test and exact McNemar coincide.
- Regressions in the (easy) direct half dilute and add noise to the pooled statistic; the paraphrase-only test is unaffected by direct regressions. Hence the pooled test needs about 0.01 to 0.02 more MRR in the regression scenarios.
- **Percentile cluster-bootstrap CI (B = 2000, 1000 datasets per scenario) [computed]:** at `q_up` = 0.5 without regressions (permutation power 0.80) the pooled CI excludes 0 in 97.5% of datasets and covers the true gain in 98%; mean width 0.063 (half-width 0.031). With one expected regression and `q_up` = 0.5: excludes 0 in 46.9% (permutation power 0.32), width 0.072. With a small true gain (`q_up` = 0.2, 1 regression, truth +0.005): coverage 0.90. So the bootstrap CI is more liberal than the permutation test and under-covers when few clusters change. Use it to display the effect size, not to decide.
- Heterogeneous interpretation: a realistic hybrid gain on paraphrase is on the order of +0.05 to +0.10 MRR; the smallest detectable gains above are at or beyond that range. Failing to reject is the likely outcome even for a real improvement of 4 to 5 fixed questions `[inference]`.

---

## 10. Recommended analysis procedure for `eval/stats.py`

### 10.1 Inputs and conventions

- One record per (question id, system): `stratum` in {direct, paraphrase, article_ref, out_of_scope}, `cluster_id`, `rank` = rank of the first correct article in the top 10 (integer 1 to 10) or `None` for a miss. Assert both systems cover identical question ids and that every in-scope cluster has exactly one direct and one paraphrase question.
- `rr(rank) = 0 if rank is None else 1/rank`; `hit1 = (rank == 1)`; `mfr(rank) = 11 if rank is None else rank` (cap 11 declared in advance).
- Difference direction: always hybrid B minus dense A for gain metrics (R@1, RR); for `mfr` use A minus B so that positive means better.
- All p-values two-sided, alpha 0.05, decision on Holm-adjusted values. Report 4 decimals, report the metric difference and its cluster-bootstrap CI beside every p.
- Constants: `SEED_PERM = 20261125`, `SEED_BOOT = 20261126`, `M_MC = 100000`, `B_BOOT = 10000`, `K_EXACT_MAX = 20`, `EPS = 1e-9` (or integer scaling of all differences by 5040 = 2 x lcm(1..10), which makes ties exact: RR differences times 5040 are integers). Use `numpy.random.default_rng(seed)`; record seeds and the numpy version in `eval/results.md`.
- Confirmatory contrasts (declare before running; at most 2): C1 = paraphrase R@1, hybrid versus dense (mid-p McNemar); C2 = paraphrase MRR@10 (45 one-question clusters, sign-flip). Recommended over pooled MRR because of dilution (9.3) `[inference]`. Everything else (pooled, direct, article_ref, MFR, sparse-only arms) is exploratory and labelled so. Count prior looks at the frozen set in the narrative.

### 10.2 Pseudocode

```
# T1  McNemar on R@1 within a stratum
def mcnemar(b, c):            # b = B hit & A miss, c = A hit & B miss
    n = b + c
    if n == 0: return exact=1.0, mid=1.0
    f  = comb(n, b) / 2**n                     # point probability of the observed count
    if b == c: return exact=1.0, mid=1.0 - f/2
    p1 = sum(comb(n, x) for x in range(min(b, c) + 1)) / 2**n
    exact = min(1.0, 2*p1)
    mid   = exact - f
    return exact, mid          # decision statistic: mid; report exact as the conservative bound

# T2  cluster sign-flip permutation test (paired)
def signflip(d):              # d[c] = mean over the cluster's questions of (B_metric - A_metric)
    nz = [x for x in d if abs(x) > EPS];  k = len(nz)
    t  = abs(sum(d))
    if k == 0: return p=1.0, method="exact"
    if k <= K_EXACT_MAX:
        sums = all 2**k values of sum(s_i * nz_i), s_i in {-1,+1}
        p = count(abs(sums) >= t - EPS) / 2**k              # observed pattern is among the 2**k
        p_mid = (count(abs(sums) > t + EPS) + 0.5*count(abs(abs(sums)-t) <= EPS)) / 2**k   # sensitivity only
        return p, "exact"
    rng = default_rng(SEED_PERM)
    S   = rng.integers(0, 2, size=(M_MC, k)) * 2 - 1
    b   = count(abs(S @ nz) >= t - EPS)
    return (b + 1) / (M_MC + 1), "mc"                       # Phipson-Smyth; identity counted via the +1

# T3  cluster bootstrap CI for the mean of d (display only)
def boot_ci(d, level=0.95):
    rng = default_rng(SEED_BOOT)
    idx = rng.integers(0, len(d), size=(B_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return quantile(means, [0.025, 0.975])                   # numpy default linear interpolation

# T4  Holm step-down over the confirmatory family (m <= 2)
def holm(ps):
    order = argsort(ps); run = 0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (m - j) * ps[i])); adj[i] = run
    return adj

# T5  regression count (direct stratum): Clopper-Pearson 95% interval for c / n_hit_in_baseline
```

Statistic definitions: C1 uses T1 on the 45 paraphrase questions. C2 uses T2 with `d` = RR_B - RR_A per paraphrase question. Exploratory pooled analyses use `d_c = 0.5 * ((RR_B - RR_A)_direct + (RR_B - RR_A)_paraphrase)` per cluster. MFR sensitivity uses T2 on `0.5 * ((mfr_A - mfr_B)_direct + (mfr_A - mfr_B)_paraphrase)`.

Reporting block per contrast: counts (b, c, n), exact p, mid-p, Holm-adjusted p (C1, C2 only), effect (R@1 count difference, MRR difference), bootstrap CI, method (exact or mc), seeds. For regressions: Clopper-Pearson upper bound on the regression rate (0 of 44 direct hits lost gives an upper limit of 8.0%; 0 of 32 paraphrase hits lost gives 10.9% [computed]), which states honestly how little "no regression" shows.

### 10.3 Test vectors (all computed; assert exactly unless marked MC)

**V1 McNemar (b, c) -> (exact two-sided, mid-p).** Swapping b and c gives the same values.

| (b, c) | exact | mid-p |
|---|---|---|
| (0, 0) | 1.0 | 1.0 |
| (5, 0) | 0.0625 | 0.03125 |
| (6, 0) | 0.03125 | 0.015625 |
| (2, 0) | 0.5 | 0.25 |
| (3, 3) | 1.0 | 0.84375 |
| (8, 1) | 0.0390625 | 0.021484375 |
| (1, 7) | 0.0703125 | 0.0390625 (Fagerland Table 6) |
| (6, 16) | 0.0524787903 | 0.0346896648 (Fagerland Table 6: 0.0525, 0.0347) |
| (10, 3) | 0.0922851562 | 0.0573730469 |
| (12, 4) | 0.0768127441 | 0.0490417480 |

Also: exact p matches `scipy.stats.binomtest(b, b+c, 0.5).pvalue` for all rows with `b + c > 0`.

**V2 sign-flip, exact branch** (`d` -> p exact, p mid):

| d | T = sum(d) | exact | mid (sensitivity) |
|---|---|---|---|
| [1,1,1,1,1] | 5 | 0.0625 | 0.03125 |
| [1,1,1,1,1,1] | 6 | 0.03125 | 0.015625 |
| [0.5, -0.5] | 0 | 1.0 | 0.75 |
| [1,1,1,-1] | 2 | 0.625 | 0.375 |
| [0.5,0.5,0.25,-0.25,0,0] | 1.0 | 0.375 | 0.25 |
| [0.5]x8 + [-0.5]x2 + [0]x20 | 3 | 0.109375 | not computed |
| [0.5]x8 + [-0.25]x2 + [0]x5 | 3.5 | 0.0234375 | not computed |
| [0,0,0] | 0 | 1.0 | 1.0 |

**V3 sign-flip, MC branch (k = 24 > 20).** `d = [0.5]*17 + [-0.5]*3 + [0.25]*4`, T = 8.0, exact p by full enumeration = 0.00042510032653808594. Assert `abs(p_mc - 0.000425100327) < 4e-4` with `M_MC = 100000` (three seeds gave 0.00056, 0.00038, 0.00045 with numpy 2.3.4; I did not verify that the integer stream is identical across numpy versions, so do not assert exact MC values). Also assert `p_mc >= 1/(M_MC+1)` and that `p_mc` is a multiple of `1/(M_MC+1)`.

**V4 Phipson convention** `(b+1)/(m+1)`: (b=0, m=9999) -> 0.0001; (0, 99999) -> 1e-05; (49, 999) -> 0.05; (500, 9999) -> 0.0501; (9999, 9999) -> 1.0. The function never returns 0.

**V5 Holm** (input -> output): [0.01, 0.04] -> [0.02, 0.04]; [0.03, 0.04] -> [0.06, 0.06]; [0.04, 0.01] -> [0.04, 0.02]; [0.2, 0.5, 0.01] -> [0.4, 0.5, 0.03]; [0.9, 0.001] -> [0.9, 0.002]. Output order equals input order; adjusted values are monotone in the sorted order and capped at 1.

**V6 baseline reproduction** (from the task's rank distributions; use any pairing): MRR@10 direct 0.9888889, paraphrase 0.8259259, pooled 0.9074074 (rounds to 0.907); R@1 direct 44/45, paraphrase 32/45; mean first rank (cap 11, no misses in the baseline) direct 1.0222222, paraphrase 1.4888889, pooled 1.2555556.

**V7 end-to-end scenario H1** (synthetic pairing A; use only for unit tests, the real pairing comes from `eval/questions.jsonl`).

- Baseline: `direct_A = [1]*44 + [2]`; `para_A = [2]*6 + [3]*5 + [4]*2 + [1]*32` (aligned by cluster, so the last cluster has direct 2 and paraphrase 1).
- Hybrid: `direct_B = [1]*45`; `para_B = [1,1,1,1,2,2, 1,1,1,3,3, 1,4, 2,2] + [1]*30`.
- Counts: paraphrase b = 8, c = 2 -> exact 0.109375, mid-p 0.0654296875.
- Per-cluster pooled-MRR differences, non-zero only: 0.25 x4, 0.333333 x3, 0.375, -0.25 x2, 0.25 (last cluster); times 5040 these are integers {1260, 1680, 1890, -1260}. Sum T = 2.125, mean 0.0472222222, k = 11 so the exact branch applies.
- Pooled MRR: baseline 0.9074074, hybrid 0.9546296; paraphrase MRR hybrid 0.9092593; direct 1.0.
- Pooled MRR cluster sign-flip: exact p = 0.0322265625 (mid 0.02197265625).
- Paraphrase-only MRR sign-flip: mean 0.0833333333, k = 10, exact p = 0.05078125 (mid 0.0361328125).
- Pooled R@1 cluster sign-flip: mean 0.0777777778, exact p = 0.0654296875 (mid 0.03857421875).
- Mean first rank, pooled: baseline 1.2555556, hybrid 1.1222222; sign-flip of the cluster differences (A minus B), mean 0.1333333, k = 11, exact p = 0.03125.
- Bootstrap CI of the pooled MRR mean difference, `SEED_BOOT = 20261126`, `B = 10000`: [0.0092593, 0.0888889] (numpy 2.3.4); assert only `lo < 0.0472222 < hi` and `0 <= lo` and `hi <= 0.12`.
- Holm over the two confirmatory p-values [C1 = 0.0654296875 (mid-p, paraphrase R@1), C2 = 0.05078125 (paraphrase MRR sign-flip)] -> [0.1015625, 0.1015625] (the smaller p is doubled, and the larger adjusted value cannot be below it). With a pooled-MRR p of 0.0322265625 in place of C2 the output would be [0.0654296875, 0.064453125].
- Lesson encoded in the vector: on the same data, paraphrase R@1 (mid-p 0.065), paraphrase MRR (0.051) and pooled MRR (0.032) land on different sides of 0.05. The plan must say before the run which one decides.

**V8 Clopper-Pearson 95%** (two-sided, `scipy.stats.beta`): 0 of 44 -> upper 0.0804; 0 of 32 -> upper 0.1089; 1 of 44 -> [0.0006, 0.1202]; 0 of 45 -> upper 0.0787.

**V9 degenerate bootstrap:** `boot_ci([0.1]*10) == (0.1, 0.1)`; `boot_ci` with a fixed seed is deterministic across calls in one environment.

### 10.4 What the code must not do

- Never return p = 0 from a sampled test (use `+1`).
- Never use Wilcoxon, plain t on binary per-question data, or the asymptotic McNemar.
- Never pool direct and paraphrase for a confirmatory claim.
- Never choose the metric, alpha, k or tokenization after seeing the frozen-set results.

---

## NEED DOWNLOAD

1. **Voorhees, E. M. and Buckley, C. (2002). "The effect of topic set size on retrieval experiment error." Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '02), Tampere, Finland, pp. 316-323.**
   - DOI: 10.1145/564376.564432. Official URL: https://doi.org/10.1145/564376.564432 (NIST record: https://www.nist.gov/publications/effect-topic-set-size-retrieval-experiment-error).
   - Reason: ACM returned HTTP 403 to an automated fetch; OpenAlex and Semantic Scholar both report no open-access copy; the NIST page offers an access-by-e-mail route only.
   - Needed: the figures/tables of error rate versus topic set size (up to 25, extrapolated to 50) and versus absolute difference in MAP; the error-rate definition and the extrapolation method; any statement on which differences are safe at about 50 topics. Used as `[inference]` input to section 1 and the "swap-rate" check.

2. **[Resolved 2026-11-25: free copy at http://sigir.org/wp-content/uploads/2020/06/p14.pdf, read in full; see section 6.]** **Sakai, T. (2020). "On Fuhr's guideline for IR evaluation." ACM SIGIR Forum 54(1), June 2020.**
   - DOI: 10.1145/3451964.3451976. Official URL: https://doi.org/10.1145/3451964.3451976. (Crossref gives pages 1-8; Ferrante et al. cite it as 14:1-14:8.)
   - Reason: OpenAlex reports closed access; no open PDF found.
   - Needed: his arguments on means of ordinal scores and on RR; the agreement rates between assessors and metrics; his remarks on multiple testing and test-collection reuse. Currently known only through Ferrante et al. and Moffat.

3. **Field, C. A. and Welsh, A. H. (2007). "Bootstrapping clustered data." Journal of the Royal Statistical Society: Series B 69(3), 369-390.**
   - DOI: 10.1111/j.1467-9868.2007.00593.x. Official URL: https://doi.org/10.1111/j.1467-9868.2007.00593.x (ANU repository, request-a-copy only, restricted until 2037: https://openresearch.anu.edu.au/items/f0a8b9ca-dd75-475f-b7f2-2f32e328c942/full).
   - Reason: publisher paywall; repository copy restricted.
   - Needed: the consistency results for the cluster bootstrap (resampling whole clusters) versus the residual bootstrap, and any note on few clusters; this supports the choice of resampling the 45 clusters.

4. **Davison, A. C. and Hinkley, D. V. (1997). Bootstrap Methods and their Application. Cambridge University Press.**
   - DOI: 10.1017/CBO9780511802843. Official URL: https://doi.org/10.1017/CBO9780511802843.
   - Reason: commercial book; no open copy.
   - Needed: the chapter on hierarchical or clustered data (cluster resampling, the two-stage bootstrap) and the section on permutation tests with the `(b+1)/(m+1)` p-value.

5. **Zuccon, G., Palotti, J. and Hanbury, A. (2016). "Query Variations and their Effect on Comparing Information Retrieval Systems." Proceedings of the 25th ACM International Conference on Information and Knowledge Management (CIKM '16), pp. 691-700.**
   - DOI: 10.1145/2983323.2983723. Official URL: https://doi.org/10.1145/2983323.2983723.
   - Reason: no open PDF found by search; the TU Wien catalogue entry returned 404.
   - Needed: how they aggregate variations per topic and whether they test significance at the topic level (mean-variance framework, section on comparing systems).

6. **Bruch, S., Gai, S. and Ingber, A. (2023/2024). "An Analysis of Fusion Functions for Hybrid Retrieval." ACM Transactions on Information Systems.**
   - DOI: 10.1145/3596512. Official URL: https://doi.org/10.1145/3596512 (PDF https://dl.acm.org/doi/pdf/10.1145/3596512).
   - Reason: OpenAlex lists it as bronze open access, but ACM answered 403 to automated fetches. The arXiv v1 and v2 texts were read instead.
   - Needed: volume/issue/year, and whether the "Interpretability and Sample Efficiency" text and Fig. 12 differ from arXiv v2 (an absolute number of tuning queries, a different step size).
   - **Update 2026-11-25:** Crossref (`https://api.crossref.org/works/10.1145/3596512`) gives volume 42, issue 1, pages 1-35, published online 2023-08-18, issue dated January 2024. The TOIS text is still unread; not needed for the Week 4 plan, which does not tune alpha.

7. **Bailey, P., Moffat, A., Scholer, F. and Thomas, P. (2015). "User Variability and IR System Evaluation." SIGIR '15.**
   - DOI: 10.1145/2766462.2767728. Official URL: https://doi.org/10.1145/2766462.2767728.
   - Reason: only the abstract page was readable.
   - Needed: the ANOVA of topic, system and query-variation factors, and any statement on how to aggregate variations before a significance test.
   - **Update 2026-11-25:** OpenAlex lists a green copy in the University of Melbourne repository (`https://hdl.handle.net/11343/58275`, marked "Open Access"), but its file list failed to load for an automated fetch. Try it in a browser before treating the paper as closed. Items 1 and 5 were re-checked the same day: OpenAlex `closed`, no repository copy.
