# Final post-hoc supplementary analyses

This post-hoc supplement reports internal same-center, same-device, same-period random-holdout evidence. It is not external or temporal validation, does not establish deployment readiness or clinical benefit, and does not support causal interpretation.

High-grade gastric lesion: Yes = HGIN + EGC + AGC; No = HC + BGL + LGIN. A model score is not a gastric-cancer probability.

The enriched diagnostic case-control/two-gate sampling design limits transportability to target clinical populations and does not establish population-screening performance.

LGIN findings remain exploratory because biopsy can undergrade disease and complete resection pathology was unavailable.

Analyses 1-4 are post-hoc model-robustness analyses. Cohort flow, included-versus-excluded descriptions, calibration-source generation, the reference-standard/timing audit, and the QC6 lineage correction are non-model reporting-closure artifacts, not a fifth model analysis.

Breath collection followed diagnostic gastroscopy and indicated biopsy and preceded treatment. Pre-endoscopy triage remains a future validation target. This study does not directly validate that use. Differential reference standards are retained as a limitation.

Session 7 integrity audit is pending; its authoritative status path is `outputs/analysis/posthoc_final_integrity_audit.json`.

Visualization release remains unauthorized and pending independent approval; its authoritative status path is `outputs/analysis/posthoc_final_visualization_release.json`.

## `outputs/analysis/posthoc_final_locked_score_metrics_minirocket_ridge.csv`

| analysis_id | view | population_id | metric | estimate | ci_low | ci_high | ci_level | ci_method | n | n_positive | n_negative | tp | tn | fp | fn | sensitivity_unadjusted | specificity_unadjusted | threshold_p_raw | score_scale | zero_cell_correction | correction_applied | estimability_status | nonestimable_reason | ci_status | ci_nonestimable_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | auc_roc | 0.9069153225806451 | 0.8794356231947932 | 0.934395021966497 | 0.95 | delong_normal_clipped | 475 | 155 | 320 |  |  |  |  |  |  |  | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | sensitivity | 0.9741935483870968 | 0.9355314540311528 | 0.9899196283965077 | 0.95 | wilson | 475 | 155 | 320 | 151 | 250 | 70 | 4 | 0.9741935483870968 | 0.78125 | 0.3418938786147648 | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | specificity | 0.78125 | 0.7327656244034267 | 0.8230619112539616 | 0.95 | wilson | 475 | 155 | 320 | 151 | 250 | 70 | 4 | 0.9741935483870968 | 0.78125 | 0.3418938786147648 | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | lr_positive | 4.453456221198157 | 3.614820374191898 | 5.486655009396517 | 0.95 | log_normal_wald | 475 | 155 | 320 | 151 | 250 | 70 | 4 | 0.9741935483870968 | 0.78125 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | lr_negative | 0.03303225806451613 | 0.012534617428611193 | 0.08704933190463444 | 0.95 | log_normal_wald | 475 | 155 | 320 | 151 | 250 | 70 | 4 | 0.9741935483870968 | 0.78125 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |
| analysis_1_difficult_spectrum | test | difficult_spectrum | auc_roc | 0.9015594541910331 | 0.8485277009735226 | 0.9545912074085436 | 0.95 | delong_normal_clipped | 119 | 38 | 81 |  |  |  |  |  |  |  | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_1_difficult_spectrum | test | difficult_spectrum | sensitivity | 0.8947368421052632 | 0.7586947827711268 | 0.9582974265244648 | 0.95 | wilson | 119 | 38 | 81 | 34 | 65 | 16 | 4 | 0.8947368421052632 | 0.8024691358024691 | 0.3418938786147648 | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_1_difficult_spectrum | test | difficult_spectrum | specificity | 0.8024691358024691 | 0.7029560930488834 | 0.8745917318129888 | 0.95 | wilson | 119 | 38 | 81 | 34 | 65 | 16 | 4 | 0.8947368421052632 | 0.8024691358024691 | 0.3418938786147648 | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_1_difficult_spectrum | test | difficult_spectrum | lr_positive | 4.529605263157895 | 2.8816210618666562 | 7.120063117090417 | 0.95 | log_normal_wald | 119 | 38 | 81 | 34 | 65 | 16 | 4 | 0.8947368421052632 | 0.8024691358024691 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |
| analysis_1_difficult_spectrum | test | difficult_spectrum | lr_negative | 0.1311740890688259 | 0.05158751128075142 | 0.33354277451753134 | 0.95 | log_normal_wald | 119 | 38 | 81 | 34 | 65 | 16 | 4 | 0.8947368421052632 | 0.8024691358024691 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | auc_roc | 0.9202607547584993 | 0.8963206957833824 | 0.9442008137336163 | 0.95 | delong_normal_clipped | 542 | 244 | 298 |  |  |  |  |  |  |  | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | sensitivity | 0.9590163934426229 | 0.9262149211658895 | 0.9775886678940718 | 0.95 | wilson | 542 | 244 | 298 | 234 | 242 | 56 | 10 | 0.9590163934426229 | 0.8120805369127517 | 0.3418938786147648 | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | specificity | 0.8120805369127517 | 0.7638600500571098 | 0.8523574858861267 | 0.95 | wilson | 542 | 244 | 298 | 234 | 242 | 56 | 10 | 0.9590163934426229 | 0.8120805369127517 | 0.3418938786147648 | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | lr_positive | 5.1033372365339575 | 4.024702684310307 | 6.4710496632019145 | 0.95 | log_normal_wald | 542 | 244 | 298 | 234 | 242 | 56 | 10 | 0.9590163934426229 | 0.8120805369127517 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | lr_negative | 0.050467416339249414 | 0.027437593414283368 | 0.09282738735508964 | 0.95 | log_normal_wald | 542 | 244 | 298 | 234 | 242 | 56 | 10 | 0.9590163934426229 | 0.8120805369127517 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |
| analysis_4_strict_qc5 | test | strict_qc5_pass | auc_roc | 0.9188505747126436 | 0.871068578280622 | 0.9666325711446653 | 0.95 | delong_normal_clipped | 133 | 58 | 75 |  |  |  |  |  |  |  | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_4_strict_qc5 | test | strict_qc5_pass | sensitivity | 0.896551724137931 | 0.7921189700751576 | 0.9517185925763711 | 0.95 | wilson | 133 | 58 | 75 | 52 | 63 | 12 | 6 | 0.896551724137931 | 0.84 | 0.3418938786147648 | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_4_strict_qc5 | test | strict_qc5_pass | specificity | 0.84 | 0.7408329231474533 | 0.9060348639642386 | 0.95 | wilson | 133 | 58 | 75 | 52 | 63 | 12 | 6 | 0.896551724137931 | 0.84 | 0.3418938786147648 | p_raw | not_applicable |  | estimable |  | estimable |  |
| analysis_4_strict_qc5 | test | strict_qc5_pass | lr_positive | 5.6034482758620685 | 3.311851500645907 | 9.480688543595008 | 0.95 | log_normal_wald | 133 | 58 | 75 | 52 | 63 | 12 | 6 | 0.896551724137931 | 0.84 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |
| analysis_4_strict_qc5 | test | strict_qc5_pass | lr_negative | 0.12315270935960591 | 0.05736179826795253 | 0.26440227260247856 | 0.95 | log_normal_wald | 133 | 58 | 75 | 52 | 63 | 12 | 6 | 0.896551724137931 | 0.84 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |
| frozen_primary_internal_test_lr | test | frozen_primary_internal_test | lr_positive | 6.373475609756098 | 4.025394404003725 | 10.091232627479519 | 0.95 | log_normal_wald | 195 | 82 | 113 | 74 | 97 | 16 | 8 | 0.9024390243902439 | 0.8584070796460177 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |
| frozen_primary_internal_test_lr | test | frozen_primary_internal_test | lr_negative | 0.11365350766909732 | 0.05859368147239798 | 0.22045243584113247 | 0.95 | log_normal_wald | 195 | 82 | 113 | 74 | 97 | 16 | 8 | 0.9024390243902439 | 0.8584070796460177 | 0.3418938786147648 | p_raw | none | False | estimable |  | estimable |  |

## `outputs/analysis/posthoc_final_locked_score_composition_minirocket_ridge.csv`

| analysis_id | view | population_id | disease_category_code | disease_category_label | outcome_label | n | denominator_n | proportion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | 0 | HC | negative | 0 | 475 | 0.0 |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | 1 | BGL | negative | 281 | 475 | 0.5915789473684211 |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | 2 | LGIN | negative | 39 | 475 | 0.08210526315789474 |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | 3 | HGIN | positive | 57 | 475 | 0.12 |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | 4 | EGC | positive | 98 | 475 | 0.2063157894736842 |
| analysis_1_difficult_spectrum | oof | difficult_spectrum | 5 | AGC | positive | 0 | 475 | 0.0 |
| analysis_1_difficult_spectrum | test | difficult_spectrum | 0 | HC | negative | 0 | 119 | 0.0 |
| analysis_1_difficult_spectrum | test | difficult_spectrum | 1 | BGL | negative | 71 | 119 | 0.5966386554621849 |
| analysis_1_difficult_spectrum | test | difficult_spectrum | 2 | LGIN | negative | 10 | 119 | 0.08403361344537816 |
| analysis_1_difficult_spectrum | test | difficult_spectrum | 3 | HGIN | positive | 14 | 119 | 0.11764705882352941 |
| analysis_1_difficult_spectrum | test | difficult_spectrum | 4 | EGC | positive | 24 | 119 | 0.20168067226890757 |
| analysis_1_difficult_spectrum | test | difficult_spectrum | 5 | AGC | positive | 0 | 119 | 0.0 |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | 0 | HC | negative | 82 | 542 | 0.15129151291512916 |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | 1 | BGL | negative | 188 | 542 | 0.34686346863468637 |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | 2 | LGIN | negative | 28 | 542 | 0.05166051660516605 |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | 3 | HGIN | positive | 38 | 542 | 0.07011070110701106 |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | 4 | EGC | positive | 75 | 542 | 0.13837638376383765 |
| analysis_4_strict_qc5 | oof | strict_qc5_pass | 5 | AGC | positive | 131 | 542 | 0.24169741697416974 |
| analysis_4_strict_qc5 | test | strict_qc5_pass | 0 | HC | negative | 19 | 133 | 0.14285714285714285 |
| analysis_4_strict_qc5 | test | strict_qc5_pass | 1 | BGL | negative | 48 | 133 | 0.3609022556390977 |
| analysis_4_strict_qc5 | test | strict_qc5_pass | 2 | LGIN | negative | 8 | 133 | 0.06015037593984962 |
| analysis_4_strict_qc5 | test | strict_qc5_pass | 3 | HGIN | positive | 9 | 133 | 0.06766917293233082 |
| analysis_4_strict_qc5 | test | strict_qc5_pass | 4 | EGC | positive | 17 | 133 | 0.12781954887218044 |
| analysis_4_strict_qc5 | test | strict_qc5_pass | 5 | AGC | positive | 32 | 133 | 0.24060150375939848 |

## `outputs/analysis/posthoc_final_qc5_correctness_minirocket_ridge.csv`

| analysis_id | view | record_type | strict_status | category_type | category | numerator_n | denominator_n | proportion | correct_n | incorrect_n | estimate | ci_low | ci_high | ci_level | ci_method | correction_applied | threshold_p_raw | score_scale | estimability_status | nonestimable_reason | ci_status | ci_nonestimable_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analysis_4_strict_qc5 | oof | accuracy | strict_pass | not_applicable | not_applicable |  | 542 |  | 476 | 66 | 0.8782287822878229 | 0.848003977540914 | 0.9031298789562027 | 0.95 | wilson |  | 0.3418938786147648 | p_raw | estimable |  | estimable |  |
| analysis_4_strict_qc5 | oof | accuracy | strict_fail | not_applicable | not_applicable |  | 235 |  | 205 | 30 | 0.8723404255319149 | 0.8236085220605613 | 0.9090950917260974 | 0.95 | wilson |  | 0.3418938786147648 | p_raw | estimable |  | estimable |  |
| analysis_4_strict_qc5 | oof | accuracy_difference | not_applicable | not_applicable | not_applicable |  |  |  |  |  | 0.005888356755907931 | -0.04169782095395384 | 0.06061369852029822 | 0.95 | newcombe |  | 0.3418938786147648 | p_raw | estimable |  | estimable |  |
| analysis_4_strict_qc5 | oof | outcome_composition | strict_pass | outcome | negative | 298 | 542 | 0.5498154981549815 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | outcome_composition | strict_pass | outcome | positive | 244 | 542 | 0.45018450184501846 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_pass | disease_category | HC | 82 | 542 | 0.15129151291512916 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_pass | disease_category | BGL | 188 | 542 | 0.34686346863468637 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_pass | disease_category | LGIN | 28 | 542 | 0.05166051660516605 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_pass | disease_category | HGIN | 38 | 542 | 0.07011070110701106 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_pass | disease_category | EGC | 75 | 542 | 0.13837638376383765 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_pass | disease_category | AGC | 131 | 542 | 0.24169741697416974 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | outcome_composition | strict_fail | outcome | negative | 148 | 235 | 0.6297872340425532 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | outcome_composition | strict_fail | outcome | positive | 87 | 235 | 0.3702127659574468 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_fail | disease_category | HC | 44 | 235 | 0.18723404255319148 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_fail | disease_category | BGL | 93 | 235 | 0.39574468085106385 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_fail | disease_category | LGIN | 11 | 235 | 0.04680851063829787 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_fail | disease_category | HGIN | 19 | 235 | 0.08085106382978724 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_fail | disease_category | EGC | 23 | 235 | 0.09787234042553192 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | oof | disease_composition | strict_fail | disease_category | AGC | 45 | 235 | 0.19148936170212766 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | accuracy | strict_pass | not_applicable | not_applicable |  | 133 |  | 115 | 18 | 0.8646616541353384 | 0.7962021511881133 | 0.9126473512649472 | 0.95 | wilson |  | 0.3418938786147648 | p_raw | estimable |  | estimable |  |
| analysis_4_strict_qc5 | test | accuracy | strict_fail | not_applicable | not_applicable |  | 62 |  | 56 | 6 | 0.9032258064516129 | 0.8045118138769257 | 0.9548881279768061 | 0.95 | wilson |  | 0.3418938786147648 | p_raw | estimable |  | estimable |  |
| analysis_4_strict_qc5 | test | accuracy_difference | not_applicable | not_applicable | not_applicable |  |  |  |  |  | -0.038564152316274525 | -0.12432952416968372 | 0.07119503617858429 | 0.95 | newcombe |  | 0.3418938786147648 | p_raw | estimable |  | estimable |  |
| analysis_4_strict_qc5 | test | outcome_composition | strict_pass | outcome | negative | 75 | 133 | 0.5639097744360902 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | outcome_composition | strict_pass | outcome | positive | 58 | 133 | 0.43609022556390975 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_pass | disease_category | HC | 19 | 133 | 0.14285714285714285 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_pass | disease_category | BGL | 48 | 133 | 0.3609022556390977 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_pass | disease_category | LGIN | 8 | 133 | 0.06015037593984962 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_pass | disease_category | HGIN | 9 | 133 | 0.06766917293233082 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_pass | disease_category | EGC | 17 | 133 | 0.12781954887218044 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_pass | disease_category | AGC | 32 | 133 | 0.24060150375939848 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | outcome_composition | strict_fail | outcome | negative | 38 | 62 | 0.6129032258064516 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | outcome_composition | strict_fail | outcome | positive | 24 | 62 | 0.3870967741935484 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_fail | disease_category | HC | 13 | 62 | 0.20967741935483872 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_fail | disease_category | BGL | 23 | 62 | 0.3709677419354839 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_fail | disease_category | LGIN | 2 | 62 | 0.03225806451612903 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_fail | disease_category | HGIN | 5 | 62 | 0.08064516129032258 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_fail | disease_category | EGC | 7 | 62 | 0.11290322580645161 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |
| analysis_4_strict_qc5 | test | disease_composition | strict_fail | disease_category | AGC | 12 | 62 | 0.1935483870967742 |  |  |  |  |  |  |  |  |  |  | estimable |  | not_applicable | not_applicable |

## `outputs/analysis/posthoc_final_within_date_concordance_minirocket_ridge.csv`

| analysis_id | view | estimate | ci_low | ci_high | ci_level | ci_method | n | n_positive | n_negative | n_date_groups_total | n_date_groups_comparable | n_date_groups_noncomparable | n_comparable_pairs | n_bootstrap_valid | n_bootstrap_attempts | bootstrap_seed | tie_credit | estimability_status | nonestimable_reason | ci_status | ci_nonestimable_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analysis_2_within_date_concordance | oof | 0.9200955667640032 | 0.8884029164025802 | 0.9437197284382493 | 0.95 | percentile_bootstrap_linear_quantile | 777 | 331 | 446 | 40 | 33 | 7 | 3767 | 2000 | 2000 | 42 | 0.5 | estimable |  | estimable |  |
| analysis_2_within_date_concordance | test | 0.9568965517241379 | 0.9195979899497487 | 0.98 | 0.95 | percentile_bootstrap_linear_quantile | 195 | 82 | 113 | 36 | 23 | 13 | 232 | 2000 | 2000 | 42 | 0.5 | estimable |  | estimable |  |

## `outputs/analysis/posthoc_final_date_grouped_validation_minirocket_ridge.csv`

| analysis_id | n | n_positive | n_negative | n_date_groups | n_folds | splitter | shuffle | split_seed | split_seed_status | model_seed | date_grouped_auc_roc | date_grouped_auc_ci_low | date_grouped_auc_ci_high | frozen_random_oof_auc_roc | auc_difference_date_grouped_minus_random | delta_ci_low | delta_ci_high | ci_method | n_bootstrap_valid | n_bootstrap_attempts | bootstrap_seed | tie_credit | oof_vector_sha256 | date_grouped_auc_status | date_grouped_auc_nonestimable_reason | date_grouped_auc_ci_status | date_grouped_auc_ci_nonestimable_reason | auc_difference_status | auc_difference_nonestimable_reason | auc_difference_ci_status | auc_difference_ci_nonestimable_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analysis_3_date_grouped_minirocket | 777 | 331 | 446 | 40 | 5 | StratifiedGroupKFold | False |  | not_applicable | 42 | 0.9163765190413613 | 0.8772736053248983 | 0.9435145064277004 | 0.924593228835029 | -0.008216709793667731 | -0.01977934524123698 | 0.001989308999544836 | percentile_bootstrap_linear_quantile | 2000 | 2000 | 42 | 0.5 | 08dca3b011fa58baadc197b8688de11a8524956b396e91534ac639644972bf18 | estimable |  | estimable |  | estimable |  | estimable |  |

## `outputs/analysis/posthoc_final_date_grouped_fold_audit_minirocket_ridge.csv`

| fold_id | n_fit | n_validation | n_fit_positive | n_fit_negative | n_validation_positive | n_validation_negative | n_fit_date_groups | n_validation_date_groups | date_overlap_count | calibrator_fit_feasible | validation_auc_roc | estimability_status | nonestimable_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 628 | 149 | 263 | 365 | 68 | 81 | 32 | 8 | 0 | True | 0.9050472040668119 | estimable |  |
| 2 | 628 | 149 | 267 | 361 | 64 | 85 | 32 | 8 | 0 | True | 0.9314338235294117 | estimable |  |
| 3 | 628 | 149 | 267 | 361 | 64 | 85 | 31 | 9 | 0 | True | 0.9558823529411765 | estimable |  |
| 4 | 611 | 166 | 258 | 353 | 73 | 93 | 31 | 9 | 0 | True | 0.8964501399322433 | estimable |  |
| 5 | 613 | 164 | 269 | 344 | 62 | 102 | 34 | 6 | 0 | True | 0.9074952561669829 | estimable |  |

## `cohort_flow`

| node_id | parent_node_id | display_order | stage | label | n | count_source | is_terminal | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assessed_source_cohort |  | 1 | assessed | Assessed source cohort | 1040 | frozen_sample_index_row_count | False | Root node |
| excluded_qc1_env_sensor_fault | assessed_source_cohort | 2 | exclusion | QC1 environmental sensor fault | 66 | validated_exactly_one_closed_set_exclusion_reason | True | Zero-count retained; key=qc1_env_sensor_fault |
| excluded_qc2_acquisition_truncated | assessed_source_cohort | 3 | exclusion | QC2 acquisition truncated | 2 | validated_exactly_one_closed_set_exclusion_reason | True | Zero-count retained; key=qc2_acquisition_truncated |
| excluded_qc3_unstable_baseline | assessed_source_cohort | 4 | exclusion | QC3 unstable baseline | 0 | validated_exactly_one_closed_set_exclusion_reason | True | Zero-count retained; key=qc3_unstable_baseline |
| excluded_qc4_insufficient_response | assessed_source_cohort | 5 | exclusion | QC4 insufficient response | 0 | validated_exactly_one_closed_set_exclusion_reason | True | Zero-count retained; key=qc4_insufficient_response |
| excluded_qc5_sensor_saturation | assessed_source_cohort | 6 | exclusion | QC5 sensor saturation | 0 | validated_exactly_one_closed_set_exclusion_reason | True | Zero-count retained; key=qc5_sensor_saturation |
| excluded_qc6_excessive_spikes | assessed_source_cohort | 7 | exclusion | QC6 excessive spikes | 0 | validated_exactly_one_closed_set_exclusion_reason | True | Zero-count retained; key=qc6_excessive_spikes |
| excluded_clip_length_insufficient | assessed_source_cohort | 8 | exclusion | Clip-length insufficient | 0 | validated_exactly_one_closed_set_exclusion_reason | True | Zero-count retained; key=clip_length_insufficient |
| modeling_cohort | assessed_source_cohort | 9 | modeling_cohort | Final modeling cohort | 972 | frozen_in_final_output_true | False | 1040=68+972 |
| train | modeling_cohort | 10 | split | Training set | 777 | locked_train_test_split_train_idx | True | Binding split |
| internal_test | modeling_cohort | 11 | split | Internal test set | 195 | locked_train_test_split_test_idx | True | Binding split |

## `included_excluded_summary`

| variable | level | group | n_total | n_nonmissing | n_missing | count | proportion | mean | sd | median | q1 | q3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age |  | included | 972 | 972 | 0 |  |  | 53.641975308641975 | 14.95091203771298 | 56.0 | 42.0 | 65.0 |
| sex | female | included | 972 | 972 | 0 | 360 | 0.37037037037037035 |  |  |  |  |  |
| endpoint | yes | included | 972 | 972 | 0 | 413 | 0.42489711934156377 |  |  |  |  |  |
| disease_category | HC | included | 972 | 972 | 0 | 158 | 0.16255144032921812 |  |  |  |  |  |
| disease_category | BGL | included | 972 | 972 | 0 | 352 | 0.36213991769547327 |  |  |  |  |  |
| disease_category | LGIN | included | 972 | 972 | 0 | 49 | 0.050411522633744855 |  |  |  |  |  |
| disease_category | HGIN | included | 972 | 972 | 0 | 71 | 0.07304526748971193 |  |  |  |  |  |
| disease_category | EGC | included | 972 | 972 | 0 | 122 | 0.12551440329218108 |  |  |  |  |  |
| disease_category | AGC | included | 972 | 972 | 0 | 220 | 0.22633744855967078 |  |  |  |  |  |
| missingness__age | missing | included | 972 | 972 | 0 | 0 | 0.0 |  |  |  |  |  |
| missingness__sex | missing | included | 972 | 972 | 0 | 0 | 0.0 |  |  |  |  |  |
| missingness__endpoint | missing | included | 972 | 972 | 0 | 0 | 0.0 |  |  |  |  |  |
| missingness__disease_category | missing | included | 972 | 972 | 0 | 0 | 0.0 |  |  |  |  |  |
| age |  | excluded | 68 | 68 | 0 |  |  | 55.838235294117645 | 13.691276549531482 | 59.0 | 47.25 | 64.25 |
| sex | female | excluded | 68 | 68 | 0 | 27 | 0.39705882352941174 |  |  |  |  |  |
| endpoint | yes | excluded | 68 | 68 | 0 | 33 | 0.4852941176470588 |  |  |  |  |  |
| disease_category | HC | excluded | 68 | 68 | 0 | 5 | 0.07352941176470588 |  |  |  |  |  |
| disease_category | BGL | excluded | 68 | 68 | 0 | 26 | 0.38235294117647056 |  |  |  |  |  |
| disease_category | LGIN | excluded | 68 | 68 | 0 | 4 | 0.058823529411764705 |  |  |  |  |  |
| disease_category | HGIN | excluded | 68 | 68 | 0 | 5 | 0.07352941176470588 |  |  |  |  |  |
| disease_category | EGC | excluded | 68 | 68 | 0 | 15 | 0.22058823529411764 |  |  |  |  |  |
| disease_category | AGC | excluded | 68 | 68 | 0 | 13 | 0.19117647058823528 |  |  |  |  |  |
| missingness__age | missing | excluded | 68 | 68 | 0 | 0 | 0.0 |  |  |  |  |  |
| missingness__sex | missing | excluded | 68 | 68 | 0 | 0 | 0.0 |  |  |  |  |  |
| missingness__endpoint | missing | excluded | 68 | 68 | 0 | 0 | 0.0 |  |  |  |  |  |
| missingness__disease_category | missing | excluded | 68 | 68 | 0 | 0 | 0.0 |  |  |  |  |  |
| qc_exclusion_reason | qc1_env_sensor_fault | excluded | 68 | 68 | 0 | 66 | 0.9705882352941176 |  |  |  |  |  |
| qc_exclusion_reason | qc2_acquisition_truncated | excluded | 68 | 68 | 0 | 2 | 0.029411764705882353 |  |  |  |  |  |
| qc_exclusion_reason | qc3_unstable_baseline | excluded | 68 | 68 | 0 | 0 | 0.0 |  |  |  |  |  |
| qc_exclusion_reason | qc4_insufficient_response | excluded | 68 | 68 | 0 | 0 | 0.0 |  |  |  |  |  |
| qc_exclusion_reason | qc5_sensor_saturation | excluded | 68 | 68 | 0 | 0 | 0.0 |  |  |  |  |  |
| qc_exclusion_reason | qc6_excessive_spikes | excluded | 68 | 68 | 0 | 0 | 0.0 |  |  |  |  |  |
| qc_exclusion_reason | clip_length_insufficient | excluded | 68 | 68 | 0 | 0 | 0.0 |  |  |  |  |  |

## `included_excluded_smd`

| variable | level | smd_included_minus_excluded | missingness_smd_included_minus_excluded | n_included_nonmissing | n_excluded_nonmissing | estimability_status | nonestimable_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age |  | -0.15321030158277832 | 0.0 | 972 | 68 | estimable |  |
| sex | female | -0.054902469106964016 | 0.0 | 972 | 68 | estimable |  |
| endpoint | yes | -0.1215077278214748 | 0.0 | 972 | 68 | estimable |  |
| disease_category | HC | 0.27856725340104865 | 0.0 | 972 | 68 | estimable |  |
| disease_category | BGL | -0.04182311254585088 |  | 972 | 68 | estimable |  |
| disease_category | LGIN | -0.0370257818394607 |  | 972 | 68 | estimable |  |
| disease_category | HGIN | -0.001857752958149615 |  | 972 | 68 | estimable |  |
| disease_category | EGC | -0.2533323123359978 |  | 972 | 68 | estimable |  |
| disease_category | AGC | 0.086594879257065 |  | 972 | 68 | estimable |  |

## `calibration_points`

| bin_index | n | events | nonevents | p_cal_mean | observed_fraction | observed_ci_low | observed_ci_high | ci_method | n_requested_bins | n_actual_bins | visualization_ready | estimability_status | nonestimable_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 | 0 | 20 | 0.028453625824799995 | 0.0 | 0.0 | 0.1611251580528194 | wilson | 10 | 10 | True | estimable |  |
| 1 | 19 | 0 | 19 | 0.036802309866080435 | 0.0 | 1.3877787807814457e-17 | 0.16817922405261632 | wilson | 10 | 10 | True | estimable |  |
| 2 | 20 | 0 | 20 | 0.04788654425762572 | 0.0 | 0.0 | 0.1611251580528194 | wilson | 10 | 10 | True | estimable |  |
| 3 | 19 | 3 | 16 | 0.06365785985300483 | 0.15789473684210525 | 0.05520471638534141 | 0.37565475270329085 | wilson | 10 | 10 | True | estimable |  |
| 4 | 20 | 2 | 18 | 0.1013540831691031 | 0.1 | 0.027866481213768196 | 0.3010336452284873 | wilson | 10 | 10 | True | estimable |  |
| 5 | 19 | 13 | 6 | 0.2595258988957352 | 0.6842105263157895 | 0.4601040767532747 | 0.8463562091220773 | wilson | 10 | 10 | True | estimable |  |
| 6 | 19 | 15 | 4 | 0.599907206907178 | 0.7894736842105263 | 0.5666572155618583 | 0.9149232336708375 | wilson | 10 | 10 | True | estimable |  |
| 7 | 20 | 15 | 5 | 0.8067550270179101 | 0.75 | 0.5312991223812559 | 0.8881382985923344 | wilson | 10 | 10 | True | estimable |  |
| 8 | 19 | 15 | 4 | 0.9081899733322644 | 0.7894736842105263 | 0.5666572155618583 | 0.9149232336708375 | wilson | 10 | 10 | True | estimable |  |
| 9 | 20 | 19 | 1 | 0.9503783019917066 | 0.95 | 0.7638688065532577 | 0.9911185511992044 | wilson | 10 | 10 | True | estimable |  |

## `calibration_smoother`

| grid_index | p_cal_grid | identity_observed_fraction | smoothed_observed_fraction | in_observed_support | boundary_clipped | estimator | estimability_status | nonestimable_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0 | 0.0 |  | False | False | statsmodels.nonparametric.smoothers_lowess.lowess | not_evaluated | outside_observed_support |
| 1 | 0.01 | 0.01 |  | False | False | statsmodels.nonparametric.smoothers_lowess.lowess | not_evaluated | outside_observed_support |
| 2 | 0.02 | 0.02 |  | False | False | statsmodels.nonparametric.smoothers_lowess.lowess | not_evaluated | outside_observed_support |
| 3 | 0.03 | 0.03 | 0.0 | True | True | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 4 | 0.04 | 0.04 | 0.019803400894549395 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 5 | 0.05 | 0.05 | 0.046240620576547556 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 6 | 0.06 | 0.06 | 0.072569010698489 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 7 | 0.07 | 0.07 | 0.09878216453811665 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 8 | 0.08 | 0.08 | 0.1248720777708486 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 9 | 0.09 | 0.09 | 0.15082973369394503 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 10 | 0.1 | 0.1 | 0.1766462679299889 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 11 | 0.11 | 0.11 | 0.20231388089454713 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 12 | 0.12 | 0.12 | 0.2278255405788852 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 13 | 0.13 | 0.13 | 0.25317384911241403 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 14 | 0.14 | 0.14 | 0.2783520750058706 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 15 | 0.15 | 0.15 | 0.30335466144010126 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 16 | 0.16 | 0.16 | 0.3281759455551717 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 17 | 0.17 | 0.17 | 0.3528097951879763 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 18 | 0.18 | 0.18 | 0.3772499008882807 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 19 | 0.19 | 0.19 | 0.40148787322544516 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 20 | 0.2 | 0.2 | 0.42551599520806815 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 21 | 0.21 | 0.21 | 0.4493276241805368 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 22 | 0.22 | 0.22 | 0.4729166445372307 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 23 | 0.23 | 0.23 | 0.4962764714358295 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 24 | 0.24 | 0.24 | 0.5193953078073807 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 25 | 0.25 | 0.25 | 0.5422514326568791 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 26 | 0.26 | 0.26 | 0.5648093493011321 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 27 | 0.27 | 0.27 | 0.587022284512683 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 28 | 0.28 | 0.28 | 0.6088359495543134 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 29 | 0.29 | 0.29 | 0.6301944967386206 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 30 | 0.3 | 0.3 | 0.6510434330104279 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 31 | 0.31 | 0.31 | 0.671349760273968 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 32 | 0.32 | 0.32 | 0.6911563740529927 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 33 | 0.33 | 0.33 | 0.7106929681523021 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 34 | 0.34 | 0.34 | 0.7303386365154867 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 35 | 0.35000000000000003 | 0.35000000000000003 | 0.7376456643080304 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 36 | 0.36 | 0.36 | 0.7375658812728321 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 37 | 0.37 | 0.37 | 0.7384798406197208 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 38 | 0.38 | 0.38 | 0.7360353883112123 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 39 | 0.39 | 0.39 | 0.7335896359258384 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 40 | 0.4 | 0.4 | 0.7302308131735461 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 41 | 0.41000000000000003 | 0.41000000000000003 | 0.7262448071994949 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 42 | 0.42 | 0.42 | 0.7240256302270617 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 43 | 0.43 | 0.43 | 0.7218677626782818 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 44 | 0.44 | 0.44 | 0.7217537696215807 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 45 | 0.45 | 0.45 | 0.7222091890726654 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 46 | 0.46 | 0.46 | 0.7251066280230264 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 47 | 0.47000000000000003 | 0.47000000000000003 | 0.7269325880805514 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 48 | 0.48 | 0.48 | 0.7294537068866496 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 49 | 0.49 | 0.49 | 0.7349309342559475 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 50 | 0.5 | 0.5 | 0.7414596016406664 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 51 | 0.51 | 0.51 | 0.7472856903667313 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 52 | 0.52 | 0.52 | 0.7498646894450791 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 53 | 0.53 | 0.53 | 0.7521123752437084 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 54 | 0.54 | 0.54 | 0.7546948653972789 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 55 | 0.55 | 0.55 | 0.7573788535565578 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 56 | 0.56 | 0.56 | 0.7600500303990667 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 57 | 0.5700000000000001 | 0.5700000000000001 | 0.7626714366460199 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 58 | 0.58 | 0.58 | 0.7652412597730638 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 59 | 0.59 | 0.59 | 0.7677682529037123 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 60 | 0.6 | 0.6 | 0.7702612846369212 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 61 | 0.61 | 0.61 | 0.7727282078624483 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 62 | 0.62 | 0.62 | 0.7751769981638326 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 63 | 0.63 | 0.63 | 0.7776141230832182 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 64 | 0.64 | 0.64 | 0.7800440154988307 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 65 | 0.65 | 0.65 | 0.7824692939500281 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 66 | 0.66 | 0.66 | 0.7848915653076053 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 67 | 0.67 | 0.67 | 0.7873121000520705 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 68 | 0.68 | 0.68 | 0.789732943204222 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 69 | 0.6900000000000001 | 0.6900000000000001 | 0.7921573952968265 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 70 | 0.7000000000000001 | 0.7000000000000001 | 0.794588432696884 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 71 | 0.71 | 0.71 | 0.7970284317100421 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 72 | 0.72 | 0.72 | 0.7994787537434082 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 73 | 0.73 | 0.73 | 0.8019406960713638 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 74 | 0.74 | 0.74 | 0.8044163297443493 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 75 | 0.75 | 0.75 | 0.8069078722996714 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 76 | 0.76 | 0.76 | 0.80941801486161 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 77 | 0.77 | 0.77 | 0.8119492517768353 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 78 | 0.78 | 0.78 | 0.8145041817689749 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 79 | 0.79 | 0.79 | 0.8170851156725 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 80 | 0.8 | 0.8 | 0.8196940107123093 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 81 | 0.81 | 0.81 | 0.8223329067935201 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 82 | 0.8200000000000001 | 0.8200000000000001 | 0.8250040687914117 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 83 | 0.8300000000000001 | 0.8300000000000001 | 0.8277095689328292 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 84 | 0.84 | 0.84 | 0.8304512250779987 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 85 | 0.85 | 0.85 | 0.8332304457074623 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 86 | 0.86 | 0.86 | 0.8360482239472582 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 87 | 0.87 | 0.87 | 0.8389052987721816 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 88 | 0.88 | 0.88 | 0.8418020892598009 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 89 | 0.89 | 0.89 | 0.8447394876666315 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 90 | 0.9 | 0.9 | 0.8477190774744165 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 91 | 0.91 | 0.91 | 0.8507424309447661 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 92 | 0.92 | 0.92 | 0.8538114556858903 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 93 | 0.93 | 0.93 | 0.8569279021297125 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 94 | 0.9400000000000001 | 0.9400000000000001 | 0.8600929703768436 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 95 | 0.9500000000000001 | 0.9500000000000001 | 0.8633075001234226 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 96 | 0.96 | 0.96 | 0.8665713810055615 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 97 | 0.97 | 0.97 | 0.8698834712799925 | True | False | statsmodels.nonparametric.smoothers_lowess.lowess | estimable |  |
| 98 | 0.98 | 0.98 |  | False | False | statsmodels.nonparametric.smoothers_lowess.lowess | not_evaluated | outside_observed_support |
| 99 | 0.99 | 0.99 |  | False | False | statsmodels.nonparametric.smoothers_lowess.lowess | not_evaluated | outside_observed_support |
| 100 | 1.0 | 1.0 |  | False | False | statsmodels.nonparametric.smoothers_lowess.lowess | not_evaluated | outside_observed_support |

## Reference-standard and timing facts

- RS01 [investigator_attested]: HC reference standard is based on endoscopic morphology; biopsy is generally not clinically indicated for this category.
- RS02 [not_systematically_recorded]: BGL was confirmed by a mixture of endoscopic morphology and biopsy; exact proportions are unavailable.
- RS03 [investigator_attested]: LGIN had pre-breath biopsy pathology, possible biopsy undergrading, and no complete ESD resection specimen during the study period.
- RS04 [repository_supported]: Inpatient/outpatient disposition is not a pathological gold standard.
- RS05 [investigator_attested]: HGIN, EGC, and AGC final categories were based on complete endoscopic or surgical resection pathology, which may have occurred after breath collection.
- RS06 [investigator_attested]: Clinicians responsible for routine endoscopy, pathology reporting, and treatment were unaware of future participation and breath-test results; wording is conservative result-unawareness, not a research-grade adjudication-blinding claim.
- RS07 [not_systematically_recorded]: Collector blinding was not documented.
- RS08 [repository_supported]: Breath collection occurred after diagnostic gastroscopy and indicated biopsy but before treatment.
- RS09 [not_systematically_recorded]: The exact gastroscopy/biopsy-to-breath interval distribution was not systematically recorded.
- RS10 [prohibited_claim]: No 'generally under three months' interval claim is supported.
- RS11 [prohibited_claim]: No confirmed central pathology review claim is supported.
- RS12 [repository_supported]: Differential reference standards across disease categories must be disclosed.
- RS13 [repository_supported]: The principal limitation is lack of direct pre-endoscopy-triage validation due to post-gastroscopy/indicated-biopsy sampling.

## QC6 reporting-lineage correction

# Additive QC6 reporting-lineage correction

- correction_date_utc: `2026-07-23T01:38:46Z`
- Scope: reporting closure only; this note does not edit the frozen protocol or run summary.

The operational plan used an any-channel rule: any gas-channel spike count above the configured threshold fails QC6. The executed implementation applied `not any(spike_counts > configured_threshold)` pass semantics in `src/data/qc.py::_evaluate_qc6` (commit `595d7db1525560e5e2e61936a222a71abbfe2762`). The frozen threshold was `211.0` in `configs/preprocessing.yaml` (provenance commit `421254714d7af7ba4466e1424bb443f26270d464`), so equality at 211.0 passes under the strict `>` rule.

The frozen protocol omitted the sample-level aggregation rule, while the historical frozen run summary used inaccurate six-channel quorum wording (commit `622cb3664774b3940f7bcf092004cfff62d06822`). Frozen QC artifacts from commit `650ce0e12c5f638f324ddad18f36812b14bc90ee` recorded zero QC6 failures. Consequently, this wording correction had no cohort or model impact and changes no analytical result. Future Methods and cohort-flow text must use the actually executed any-channel rule.
