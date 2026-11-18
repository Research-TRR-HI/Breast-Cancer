# AnchorShift - Breast Cancer Screening AI Research Context

**Document purpose:** A reusable project brief for future ChatGPT/Claude/Gemini sessions and solo execution.  
**Last updated:** 17 November 2026  
**Project status:** Early feasibility / novelty-validation stage. The final algorithm and novelty claim are NOT confirmed.

---

## 0. Read this first: the whole project in 60 seconds

We want to investigate whether an AI model trained on mammograms from one setting can safely identify low-risk screening examinations at a new hospital **without using that hospital's cancer labels to adjust its decision threshold**.

- **TRUST (2026)** already trains an AI to dismiss some low-risk mammograms and evaluates it on RSNA -> NLBS. However, in its external experiment it uses cancer outcomes from 20% of NLBS cases to select the final threshold, then evaluates on the remaining 80%.
- **UPA (2023)** already adjusts mammography model predictions using unlabeled images from a new acquisition domain. Therefore, "label-free adaptation to a new scanner" is NOT a new idea by itself.
- Other work already studies mammography domain generalization, including Garrucho et al. (2022), MC-SDS (MICCAI 2025), and MammoDG (IJCV 2026).
- Our current candidate question is narrower: **Can high-sensitivity, case-level screening dismissal transfer to a new hospital with a source-fixed threshold, and if it fails, can unlabeled-target adaptation improve it without hiding missed cancers?**
- The next step is NOT to invent or code a new AnchorShift architecture. First reproduce the relevant baseline and run a fair RSNA -> NLBS experiment with the threshold locked using RSNA only.

**Current verdict: MODIFY / INVESTIGATE.** Do not claim novelty or a safety guarantee yet. No experiment has been run by our team so far.

---

## 1. Project goal and clinical problem

### 1.1 What clinical task are we studying?

The task is **screening mammography triage**, not autonomous cancer diagnosis.

A screening mammogram is an X-ray examination of the breasts. A model assigns each image a cancer-suspicion score. The scores from all images belonging to one examination are combined into a case-level score. A decision threshold is then used to decide whether the examination must go to a radiologist.

- If the case score is at or above the threshold: keep it for human/radiologist review.
- If the case score is below the threshold: the model proposes that the examination can be dismissed from manual review.

This is called **selective dismissal / rule-out / selective triage**. It does not mean the AI has proven that the patient is cancer-free. It means the AI proposes that the examination is sufficiently low risk to skip a specified review step.

### 1.2 Important terms

| Term | Plain-English meaning |
|---|---|
| Mammogram | Breast X-ray image. |
| Screening examination / case | The patient's screening visit, usually containing multiple images/views. |
| Model score | The AI's numerical suspicion score. It is not automatically a true cancer probability. |
| Threshold / cutoff | The score boundary used to decide whether a case is sent for review or dismissed. |
| Dismissal rate | Percentage of all screening cases the model would remove from manual review. |
| Cancer sensitivity / triage recall | Percentage of cancer-positive cases that are NOT dismissed; they remain for review. |
| Missed cancer among dismissed cases | A cancer-positive case that the model would have dismissed. This is the central safety failure. |
| False-positive screening case | A case initially considered suspicious by the screening reader but later shown not to be cancer. |
| Domain shift | The new hospital's data differ from the training hospital's data. Causes include scanner, image processing, patient population, disease prevalence, and clinical workflow. |
| Target-site label | The new hospital's known outcome, such as confirmed cancer yes/no. |
| Unlabeled target image | A new-hospital mammogram for which we have the image but do not use its cancer outcome. |
| Test-time adaptation (TTA) | Updating a model or its prediction behavior using incoming target images during deployment, often without labels. |
| Domain generalization (DG) | Training a model so that it transfers to unseen domains, ideally without needing target data during training. |
| Calibration | Making scores/thresholds correspond to a chosen operating point or observed risk. |
| OOD | Out-of-distribution: data that differ from the data used to train or calibrate the model. |
| Conformal risk control | A statistical method for controlling a specified error/risk under assumptions such as exchangeability. It does not guarantee safety under arbitrary unknown shifts. |

### 1.3 Three different meanings of "no target labels"

Do not mix these protocols. They answer different questions.

1. **Strict source-only / frozen-threshold transfer:** Do not use target images for adaptation or threshold selection. Target images are used only for inference. The source-trained model and source-selected threshold are frozen before the target hospital is tested.
2. **Unsupervised target adaptation:** Use unlabeled images from the new hospital to adapt predictions or model parameters, but do not use cancer outcomes. UPA is an example. This is label-free, but it is not zero-target-data.
3. **Target-label calibration:** Use some new-hospital images with known cancer outcomes to select or adjust the threshold. This is what TRUST does in its RSNA -> NLBS external experiment.

Our first experiment must test protocol 1. A second experiment can test protocol 2. Protocol 3 can be reported as an upper/reference condition, but it must never be described as zero-label deployment.

---

## 2. Working project: AnchorShift

**Working codename:** AnchorShift - Safe Screening Across Unseen Hospitals.

The name is provisional. It is not yet a finalized algorithm or a confirmed novelty claim.

### 2.1 Current research question

> Can a mammography AI model trained and thresholded using source-hospital data maintain high-sensitivity, case-level dismissal when transferred to a previously unseen screening hospital without using that hospital's cancer labels? If not, can adaptation using only unlabeled target images improve the operating point, and under what types of shift does it fail?

### 2.2 Why this matters

A score threshold that works at one hospital may not work at another. Scanner manufacturer, acquisition settings, image processing, patient age/density, cancer prevalence, and clinical population can change the model's score distribution or its ability to separate cancer from non-cancer.

A model can retain a similar AUROC (ranking quality) but still have a different sensitivity at the fixed threshold. That is dangerous for dismissal: some cancer cases could fall below the threshold and be incorrectly removed from review.

### 2.3 What we are NOT claiming

- We are not claiming to be the first to do mammography domain generalization.
- We are not claiming to be the first to adapt mammography predictions without target labels.
- We are not claiming to be the first to build AI-based mammography dismissal/triage.
- We are not claiming a universal or mathematically guaranteed 95%/98% sensitivity at any unseen hospital.
- We are not claiming that anatomical symmetry, CC/MLO consistency, test-time adaptation, or conformal prediction is novel by itself.

The candidate contribution, if the experiments justify it, must be a specific improvement in **high-sensitivity case-level dismissal under a clearly defined unseen-site shift**, compared fairly with existing methods.

---

## 3. What TRUST (2026) actually did

**Paper:** *TRUST: Threshold-Recalibrated Uncertainty-Safe Training for Certified Dismissal in Breast Cancer Screening*  
**arXiv:** https://arxiv.org/abs/2609.00300  
**Uploaded source in this project:** `2609.00300v1.pdf` (25 pages).  
**Important sections:** Methods 3.1-3.11; Results 4.1-4.3; Discussion 5.4 and Limitations 5.8.

### 3.1 Model

The uploaded TRUST paper uses:

- EfficientNet-B5 image encoder initialized with Mammo-CLIP pretrained weights.
- The image encoder is frozen; only the classification head is trained.
- The head has LayerNorm, dropout 0.3, and one linear output followed by sigmoid.
- The model gives a score per mammogram image.
- For a screening case, the case score is the **maximum image score** across images in that case. If any image is suspicious, the whole case remains for review.

This is not a learned four-view fusion transformer. It is image-level scoring followed by max aggregation at case level.

### 3.2 Training objective

TRUST combines:

- Binary cross-entropy loss (ordinary cancer/non-cancer classification loss).
- Focal loss (gives more attention to hard examples / class imbalance).
- A dismissal loss that penalizes cancer-positive images whose score is too close to or below the current dismissal boundary.

The dismissal term is:

```text
L_dismiss = (1 / N_positive) * sum over positive images i of max(0, tau_t + margin - p_i)
```

Here `p_i` is the model score, `tau_t` is a provisional threshold recalculated during training, and `margin` keeps positive cases away from the dismissal region.

TRUST recalculates `tau_t` each epoch using a separate internal training-calibration subset. In the paper, the in-domain cross-validation setup uses 70% for model fitting, 10% for training calibration, and 20% outer test per fold. The final operating threshold is selected separately after out-of-fold predictions are generated.

### 3.3 Datasets used by TRUST

- **RSNA:** 11,913 training examinations, one examination per participant; the TRUST preprocessing pipeline reports 54,706 images and 486 cancer-positive examinations (about 4.1% positive in its processed cohort). The data include Emory Healthcare (USA) and BreastScreen Victoria (Australia).
- **NLBS:** 5,997 screening cases, 26,988 images, 149 cancer-positive cases. Images are from GE Senographe Essential systems. Cancer-positive cases were confirmed through further diagnostic procedures/laboratory analysis; false-positive screening cases were later shown to be nonmalignant; normal cases had no interval cancer over at least two years.

### 3.4 The key external-evaluation detail

This is the most important fact for our project.

In TRUST Section 3.11 and Section 4.3:

1. Train the final model using RSNA. Model weights are not retrained on NLBS.
2. Apply the RSNA-trained model to NLBS.
3. Divide NLBS into a 20% threshold-search subset and an independent 80% evaluation subset.
4. Use the cancer outcomes from the 20% NLBS threshold-search subset to choose the operating threshold for the 95% and 98% recall targets.
5. Freeze that selected threshold and evaluate on the remaining 80% NLBS cases.

The paper states that the target threshold is selected because prediction-score distributions may differ across datasets. Therefore, TRUST's external experiment is **fixed model weights + target-label threshold calibration**. It is not a zero-target-label external experiment.

### 3.5 TRUST external results (do not misrepresent these)

For RSNA -> NLBS, after using the 20% labeled NLBS threshold-search subset:

- At the 98% target-recall operating point, TRUST dismissed 12.95% of NLBS cases; the one-sided 99% Clopper-Pearson upper bound on cancer prevalence among dismissed cases was 1.06%.
- At the 95% target-recall operating point, TRUST dismissed 19.87%; the corresponding upper bound was 1.37%.
- The CE-only baseline did not meet the 98% target in that external evaluation. At the 95% target it dismissed 13.55%, with upper bound 1.77%.

These are **not** results for a threshold frozen from RSNA. They cannot be quoted as evidence that TRUST works with zero target labels.

### 3.6 What TRUST's statistical certificate means

TRUST reports a one-sided 99% Clopper-Pearson upper confidence bound on the cancer prevalence among dismissed cases in the evaluation sample. The unit is the **case**, not the individual image, because images from the same examination are not independent.

This is a retrospective confidence bound for the evaluated sample under the binomial independence assumptions. It is NOT a guarantee that future patients at any arbitrary hospital will be safe. TRUST itself says its result is not proof of universal generalizability and that more populations and prospective workflow studies are needed.

### 3.7 Reproduction / code status

The uploaded TRUST PDF identifies the datasets but does not provide a public code repository or a TRUST checkpoint URL in its Data and Code Availability section. Do not assume pretrained TRUST weights are available. Before implementation:

1. Search the authors' GitHub/project pages and arXiv ancillary files.
2. If no implementation/checkpoint is available, reproduce the model and loss from the paper.
3. Mammo-CLIP's pretrained encoder is a separate resource; having that checkpoint is not the same as having a trained TRUST model.

**Citation metadata caution:** The uploaded PDF title page lists the first author as Parham Shafie, while the current arXiv HTML metadata lists Parham Hajishafiezahramini. Check the exact version and author metadata before citing in a manuscript.

---

## 4. Prior work we must beat or distinguish ourselves from

This is a selected set of directly relevant papers, not a claim that every paper in the world has been found.

### 4.1 UPA - the most important prior work for the zero-label question

**Roschewitz et al. (2023), Nature Communications:** *Automatic correction of performance drift under acquisition shift in medical image classification.*  
DOI: https://doi.org/10.1038/s41467-023-42396-y  
Code: https://github.com/biomedia-mira/upa

What it does:

- Uses a trained model and a reference set of source-domain predictions.
- Receives a limited set of **unlabeled** images from the new acquisition domain.
- Aligns the cumulative distribution of target predictions to the reference prediction distribution using piecewise-linear CDF matching (similar to histogram matching, but on model scores).
- Uses the aligned scores to restore a chosen sensitivity/specificity operating point under the acquisition-shift scenarios studied.
- Does not require cancer labels from the target site for this adaptation step.

Why it matters:

UPA already demonstrates label-free mammography prediction recalibration under acquisition shift. Therefore, AnchorShift cannot claim that it is the first to transfer or recalibrate mammography scores without target labels.

What UPA does NOT establish:

- It is not a TRUST-style, case-level high-recall dismissal training method with a separate dismissal-risk certificate.
- Its success depends on assumptions about how the score distributions/classes change. It is not a universal solution to arbitrary population, prevalence, manifestation, or ranking shifts.
- Its paper says its monotone alignment preserves ranking/AUROC; if the model's class separability/AUROC drops substantially in the new domain, prediction alignment cannot restore that lost ranking performance.
- The mammography datasets used in the UPA paper include OPTIMAM and internal UK/Hungary screening cohorts that are not all freely downloadable. The method code is public, but exact data replication may be access-limited.

**Implication:** UPA must be a major baseline. Our proposed experiment must determine whether UPA already solves the particular RSNA -> NLBS operating-point problem.

### 4.2 Mammography domain generalization

**Garrucho et al. (2022), Artificial Intelligence in Medicine:** *Domain generalization in deep learning based mass detection in mammography: A large-scale multi-center study.*  
DOI: https://doi.org/10.1016/j.artmed.2022.102386

- Trains single-domain mass detectors and tests on five unseen domains.
- Studies scanner, acquisition, patient, breast-density, and lesion-related shifts.
- Proposes a single-source generalization pipeline that does not require images from the new domain during training.
- Task is lesion/mass detection, not case-level safe dismissal with a high-recall operating threshold.

**MC-SDS (MICCAI 2025):** *Domain Generalization for Mammogram Classification by Suppressing Domain-Specific Features.*  
DOI: https://doi.org/10.1007/978-3-032-04981-0_17  
Open-access paper page: https://papers.miccai.org/miccai-2025/0251-Paper3338.html

- Uses an Adaptive Channel Filter (ACF) to suppress feature channels associated with domain-specific information.
- Uses a Low-Frequency Perturbation Module (LFPM) to perturb low-frequency image components and reduce reliance on acquisition-specific features.
- Evaluates mammogram classification across a public dataset and two internal datasets.
- Does not establish target-label-free, case-level high-sensitivity dismissal or a formal target-site dismissal guarantee.

**MammoDG (2026), International Journal of Computer Vision:** *MammoDG: Generalisable Deep Learning Advances Cross-Domain Multi-Center Breast Cancer Screening.*  
DOI: https://doi.org/10.1007/s11263-026-02979-w  
Open repository: https://www.repository.cam.ac.uk/items/e69106af-18ca-4eb0-8004-6837487a4022

- Uses a cross-channel cross-view enhancement module for information interaction among multi-view mammograms.
- Uses instance-level contrastive regularization to reduce multi-center distribution gaps.
- Is a strong recent comparator for cross-domain mammography screening generalization.
- Its stated contribution is generalizable mammography analysis; it does not by itself establish the specific source-locked high-sensitivity dismissal protocol we want to test.

**Dataset-Origin Signatures (2026 preprint):** *Dataset-Origin Signatures and Shortcut Learning in Screening Mammography AI: A Cross-Dataset Case Study.*  
arXiv: https://arxiv.org/abs/2607.15416

- Shows that datasets can remain highly distinguishable by origin despite consistent preprocessing.
- Reports that naively adding abnormal-enriched external cases (e.g., CBIS-DDSM and CMMD) to a screening cohort can reduce screening-cohort performance in the studied setup.
- This is evidence that cross-dataset mixing can introduce shortcuts; it is not a solution to safe dismissal.

### 4.3 Safe dismissal / threshold selection

**Shafie et al. / current arXiv metadata: TRUST (2026)** - see Section 3 above.

**Bernstein et al. (2026), PLOS Digital Health:** *A novel statistical framework for quantifying risks and benefits of AI automation in screening mammography.*  
DOI: https://doi.org/10.1371/journal.pdig.0001231

- Formalizes the trade-off between how many screening cases can be ruled out and the risk of missed cancers.
- Important for defining appropriate rule-out metrics and operating points.
- It evaluates thresholds on scores from an already-trained model; it is not itself a zero-label cross-hospital adaptation method.

### 4.4 Generic test-time adaptation and conformal risk control

These are useful methods to benchmark, but they are not mammography-specific safe-dismissal solutions by themselves.

- **TENT (Wang et al., ICLR 2021):** https://arxiv.org/abs/2006.10726. Uses entropy minimization and normalization updates on unlabeled test batches. It can adapt to shifts, but it does not automatically guarantee clinical cancer sensitivity or safe dismissal.
- **SAR (Niu et al., ICLR 2023):** https://arxiv.org/abs/2302.12400. Stabilizes test-time adaptation under mixed shifts, small batches, and imbalanced streams. Still not a mammography-specific safe-dismissal guarantee.
- **Conformal Risk Control (Angelopoulos et al., ICLR 2024):** https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html. Provides expected-risk control under its assumptions, including exchangeability-based conditions. Standard conformal guarantees do not automatically survive arbitrary unseen hospital/scanner shifts.

---

## 5. Dataset plan and what each dataset can honestly support

### 5.1 Primary experiment datasets

| Dataset | Scale / labels | Access | Best role | Important caveats |
|---|---|---|---|---|
| **RSNA Screening Mammography** | Public training partition: 11,913 examinations. TRUST preprocessing reports 54,706 images and 486 cancer-positive exams. Data from Emory (USA) and BreastScreen Victoria (Australia). Positive cases are pathology-confirmed; negative labels use follow-up/benign/diagnostic assessment. | Public for non-commercial research via Kaggle/RSNA terms. | Source training and source-only threshold calibration. | Cancer-enriched compared with ordinary screening. Use the public labeled training partition; do not assume hidden challenge-test labels are available. |
| **NLBS / NL-Breast-Screening** | 5,997 screening cases, 26,988 images: 149 cancer-positive, 1,516 false-positive, 4,332 normal. 5,935 unique women; 62 case records are repeat exams. | Open CC BY 4.0 via FRDR. | Main external screening test. | Single Canadian screening program; GE Senographe Essential systems (2008-2010). No breast-density labels or lesion boxes. Ethnicity is not recorded as a usable subgroup variable; dataset descriptor notes the population is assumed predominantly European. Group repeat exams by patient for splitting/statistics. |

Sources: RSNA dataset paper https://doi.org/10.1148/ryai.250375 ; NLBS dataset paper https://doi.org/10.1038/s41597-025-05866-0 ; NLBS paper/data details https://pmc.ncbi.nlm.nih.gov/articles/PMC12378999/ .

### 5.2 Optional datasets (do not misuse them)

| Dataset | Scale / label type | Access | Appropriate use | Why it is not the main screening dismissal test |
|---|---|---|---|---|
| **IMDB (India, 2026)** | 12,674 mammography images from 3,219 patients across IMDB r1.0/r2.0; age, density, BI-RADS; histopathology for suspicious lesions and follow-up/expert reading for lower-risk cases. | Open via IBIA and MIDAS. | Additional Indian population/domain evaluation, with label semantics carefully documented. | Diagnostic/opportunistic, screening-naive cohort; not equivalent to a population screening program. Do not claim it is a fully representative national screening cohort. |
| **CMMD (China)** | 1,775 patients; official sources report 3,728 mammographies and 5,202 DICOM image objects. Biopsy-confirmed benign/malignant tumors. | Public TCIA, CC BY 4.0. | Diagnostic lesion classification / external stress test. | Biopsy-enriched disease cohort, not a normal screening population. Not suitable for estimating a real screening dismissal rate or screening PPV. |
| **VinDr-Mammo (Vietnam)** | 5,000 exams / 20,000 images; two hospitals; three imaging vendors reported in the dataset paper; radiologist BI-RADS, density and lesion annotations. | PhysioNet restricted/credentialed access and data-use terms. | Detection, BI-RADS, density and acquisition-domain robustness. | BI-RADS/radiologist annotations are not the same as biopsy-confirmed cancer outcomes. Not a clean pathology-confirmed cancer-dismissal endpoint. |
| **EMBED (USA)** | Open-data release: 76,373 exams, 22,382 patients, 480,323 images (2D and synthetic 2D); the full resource is much larger. | Application and approval through HITI portal; 90-day access. | Potential multi-site/vendor subgroup work if access is approved. | Not an unrestricted download. The DUA restricts sharing data and public release/commercialization of models, weights, and embeddings trained using EMBED. |
| **CBIS-DDSM** | 1,566 participants; digitized film mammograms with lesion/ROI and pathology information. | Public TCIA. | Detection/ROI pretraining or legacy-domain stress test. | Digitized film is not modern FFDM; case/ROI crops are not independent screening examinations. Not a primary modern screening external cohort. |

Dataset references:

- IMDB 2026 paper: https://www.melba-journal.org/papers/2026:050.html ; DOI https://doi.org/10.59275/j.melba.2026-bd61
- CMMD TCIA: https://www.cancerimagingarchive.net/collection/cmmd/ ; DOI https://doi.org/10.7937/tcia.eqde-4b16
- VinDr-Mammo PhysioNet: https://physionet.org/content/vindr-mammo/1.0.0/ ; DOI https://doi.org/10.13026/br2v-7517
- EMBED access/terms: https://docs.hitilab.com/docs/datasets/embed/open-data-faq

### 5.3 Critical dataset limitation

For the core question of **population-screening safe dismissal**, RSNA and NLBS are the strongest immediate pair. NLBS is the key external screening cohort, but it is one screening program and one GE device family. IMDB and CMMD can add population/diagnostic stress tests but cannot replace a second independent population-screening cohort with comparable cancer outcome labels.

This is a real study limitation. Do not claim broad multi-vendor or global screening validation from RSNA -> NLBS alone.

---

## 6. What is the actual research gap right now?

### 6.1 Broad claims that are already occupied

| Broad claim | Status |
|---|---|
| Mammography AI should generalize across hospitals/scanners. | Already extensively studied. |
| Mammography AI can adapt to a new scanner using unlabeled images. | Already demonstrated by UPA (2023). |
| Mammography models can be trained for domain generalization. | Already studied by Garrucho et al. (2022), MC-SDS (2025), MammoDG (2026), and others. |
| AI can dismiss low-risk screening mammograms to reduce human workload. | Already studied by TRUST (2026) and clinical screening workflow research. |
| Conformal/selective prediction can control risk under suitable statistical assumptions. | Established general methodology; not a new contribution by itself. |

### 6.2 Narrow candidate gap (NOT YET PROVEN)

> High-sensitivity, case-level mammography dismissal under an unseen hospital shift, with the decision threshold fixed using source data only, and with no target cancer labels used for calibration. If the frozen threshold fails, determine whether unlabeled-target adaptation (especially UPA) fixes it, and identify the specific shift conditions where existing methods fail.

This is a research hypothesis, not a confirmed novelty claim. A failure on one dataset pair does not prove a new method is needed. We must compare UPA and domain-generalization baselines and investigate the cause of failure.

### 6.3 Why we cannot promise a universal guarantee

A finite-sample risk guarantee generally needs assumptions connecting calibration and test data (for example, exchangeability or a correctly specified shift model). Arbitrary unseen hospital shifts can violate those assumptions. A new hospital can have a different prevalence, patient mix, scanner, acquisition pipeline, and disease manifestation distribution.

Therefore:

- Do not claim "guaranteed 95% sensitivity at any unseen hospital."
- Do not say conformal prediction solves arbitrary domain shift.
- Do not treat TRUST's retrospective Clopper-Pearson bound as a future-hospital guarantee.
- If assumptions are not justified, report observed performance and uncertainty; defer uncertain cases rather than claiming certification.

---

## 7. The first experiment: exact protocol

**Goal:** Determine whether TRUST's source-selected threshold transfers from RSNA to NLBS without target cancer labels.

### 7.1 Stage A - obtain and audit the data

1. Download the labeled RSNA training partition under Kaggle/RSNA terms.
2. Download NLBS from FRDR under CC BY 4.0.
3. Build a metadata manifest with at least: patient/case ID, exam ID, image path, laterality, view (CC/MLO), label, dataset/source, scanner metadata where available.
4. Verify the number of unique patients, exams, images, cancer-positive cases, normal cases, and false-positive cases after preprocessing.
5. For NLBS, identify the 62 repeat case records and group by patient. Do not allow the same woman's repeated exams to leak between adaptation/calibration/evaluation partitions. If a stable patient key cannot be recovered, use one exam per woman or patient-clustered inference where possible.
6. Check label definitions and DICOM preprocessing before training.

### 7.2 Stage B - reproduce TRUST on RSNA

- Use the TRUST method as described in the uploaded paper.
- If TRUST weights/code are unavailable, reproduce the architecture, preprocessing, loss, and training protocol from the Methods section.
- Keep all source splits patient-level.
- A practical initial split is: model-fitting subset, TRUST's internal training-calibration subset (used to update its training-time provisional threshold), and a separate held-out source operating-threshold calibration subset. The exact percentages should be predeclared and reported. A 70/10/20 split is a reasonable starting template, but reproduce the paper's cross-validation design if feasible.
- The final source operating threshold must be selected using only the held-out RSNA calibration labels to achieve the chosen source target recall (for example 95% and 98%).
- Save the model checkpoint, preprocessing config, source calibration scores, selected threshold(s), code commit, and random seeds.

### 7.3 Stage C - strict zero-target-label test on NLBS

Before running the target test, freeze and record:

- Model weights.
- Preprocessing.
- Case-level aggregation rule.
- Source-selected threshold(s).
- All hyperparameters.

Then:

1. Run NLBS images through the frozen model.
2. Do not use NLBS cancer labels for training, threshold selection, hyperparameter tuning, checkpoint selection, or adaptation.
3. Save all NLBS predictions and a timestamped/hash-locked prediction file.
4. Only after predictions and decisions are frozen, reveal/use NLBS labels to calculate evaluation metrics.

This is the clean test of strict source-only transfer.

### 7.4 Stage D - compare label-free adaptation

Run UPA as a separate comparator:

- Use a designated NLBS adaptation subset containing images only (no cancer outcomes).
- Use source reference predictions from RSNA.
- Fit UPA's prediction-distribution alignment using the unlabeled NLBS adaptation images.
- Apply the fitted alignment to a separate NLBS evaluation subset.
- Keep patient identities separated between adaptation and evaluation; group repeat exams by patient.

This is **unlabeled-target adaptation**, not strict zero-shot. Report it separately from the frozen-threshold result.

If feasible, add TENT/SAR and MC-SDS/MammoDG baselines, but do not delay the first clean test trying to reproduce every paper. Ensure code, data, task and training budgets are compatible before claiming a fair comparison.

### 7.5 Required baselines

Minimum:

1. CE-only model with the same encoder/backbone and source threshold.
2. Reproduced TRUST model with a source-only threshold.
3. UPA (2023) with unlabeled NLBS alignment images and no NLBS labels.
4. TRUST's published target-calibrated external setting as a **reference-only upper comparison**, not as a zero-label baseline.

Next tier if feasible:

5. MC-SDS (MICCAI 2025) or MammoDG (IJCV 2026) for representation/domain generalization.
6. TENT/SAR for generic test-time adaptation, with safeguards against collapse under small batches and extreme class imbalance.

### 7.6 Metrics to report

Primary:

- **Cancer sensitivity / triage recall:** fraction of cancer-positive cases that remain for radiologist review.
- **Dismissal rate:** fraction of all cases that would be dismissed.
- **Missed cancer count/rate:** cancer-positive cases that fall below the dismissal threshold.
- **Cancer prevalence among dismissed cases / false omission rate:** cancer-positive dismissed cases divided by all dismissed cases.
- **One-sided confidence bound** for cancer prevalence among dismissed cases, with assumptions and unit clearly stated.

Secondary:

- AUROC and AUPRC.
- False-positive cases dismissed (screening cases initially recalled but later confirmed non-cancer).
- Risk-coverage curve (how missed-cancer risk changes as more cases are dismissed).
- Calibration metrics if scores are interpreted as probabilities; TRUST itself treats sigmoid outputs as scores, not calibrated probabilities.
- Patient-level confidence intervals / cluster bootstrap because NLBS contains repeat exams.

Do not report only AUROC. A model can retain AUROC but have a threshold that no longer achieves the intended sensitivity.

### 7.7 Interpretation / go-no-go table

| Result | What it means | Next move |
|---|---|---|
| Frozen RSNA threshold keeps target sensitivity and useful dismissal on NLBS. | This pair does not demonstrate the proposed failure. | Do not claim a gap. Test another suitable screening cohort or reconsider the project. |
| Frozen threshold loses sensitivity; UPA restores it. | UPA may already solve the observed acquisition-score shift. | Study whether the failure is only acquisition shift or whether another well-defined shift remains; do not propose a duplicate UPA. |
| Frozen threshold loses sensitivity; UPA does not restore it; AUROC remains stable. | Possible threshold/score-distribution problem beyond the tested UPA assumptions. | Analyze prevalence, class-conditional score shifts and alignment failure; candidate method may focus on this exact condition. |
| Frozen threshold loses sensitivity; AUROC also drops. | The model's class separation/representation fails, not merely its cutoff. | Need a domain-generalization/representation solution; compare against MC-SDS/MammoDG/Garrucho. |
| Sensitivity holds but dismissal rate collapses. | It remains conservative/safe in observed data but provides little workload reduction. | Investigate whether useful dismissal can be increased without increasing missed cancers. |

A poor result from TRUST alone is not proof of novelty. The failure must persist against the strongest relevant existing baselines and be reproducible.

---

## 8. Possible future method directions (not approved yet)

Only choose a method after Stage 7 shows a repeatable failure and after checking prior work.

Potential directions to investigate:

1. **Source-only robust threshold selection:** train/validate a decision rule on multiple source domains, then freeze it. This is only interesting if it improves on UPA and domain-generalization baselines.
2. **Unlabeled target score alignment with failure detection:** use unlabeled target images to adapt scores, but detect when the assumptions behind alignment are violated (for example, class separation changes or target population/prevalence shift makes score alignment unsafe).
3. **Selective deferral under detected shift:** if the system cannot support a reliable dismissal decision, send the case to a radiologist instead of forcing a negative decision. The method must show that deferral enriches for genuinely hard/risky cases and preserves useful coverage.
4. **Anatomical / multi-view signals:** contralateral symmetry and CC/MLO consistency were suggested in an earlier AnchorShift prompt. These are not currently part of the chosen experiment. They must not be called scanner-invariant without evidence: cancer can itself create asymmetry, and positioning/view quality can change between acquisitions.
5. **Conformal or risk-control layer:** only if the assumptions are explicitly stated and validated. No arbitrary-shift guarantee.

Do not simply combine TENT + UPA + conformal prediction + a symmetry module and call the combination novel. Each component needs a specific failure it solves and an ablation that demonstrates its value.

---

## 9. Immediate action list (Data & Model Tracks)

### Person A - data

- [ ] Obtain RSNA public training data and check Kaggle terms.
- [ ] Obtain NLBS DICOM and metadata from FRDR.
- [ ] Create patient/exam/image manifest.
- [ ] Verify unique patients, repeated exams, labels, view positions and DICOM metadata.
- [ ] Confirm storage/GPU requirements.

### Person B - model / paper reproduction

- [ ] Read TRUST Methods 3.1-3.11 and create an implementation checklist.
- [ ] Check whether TRUST code/checkpoints are publicly released; if not, plan reproduction.
- [ ] Obtain the Mammo-CLIP checkpoint used by TRUST, if its terms allow this use.
- [ ] Reproduce the CE-only baseline first, then TRUST loss.
- [ ] Implement source-only threshold calibration and prediction export.

### Together

- [ ] Agree on patient-level split protocol before training.
- [ ] Agree on 95% and 98% source sensitivity operating points.
- [ ] Freeze the model and source threshold before any NLBS labels are examined.
- [ ] Run the strict source-only NLBS test.
- [ ] Run UPA with unlabeled NLBS images as a separate experiment.
- [ ] Review results and decide PROCEED / MODIFY / ABANDON.

**The single next deliverable:** A reproducible table comparing CE-only, TRUST-source-threshold, and UPA-unlabeled-target on NLBS, showing sensitivity, dismissal rate, missed cancers, and confidence intervals.

---

## 10. Project guardrails / things future assistants must remember

1. Keep the explanation simple, but do not simplify away the statistical distinctions.
2. Clearly separate facts from hypotheses and proposed experiments.
3. Do not claim the gap is novel until a literature audit and baseline experiment support it.
4. Do not claim a universal safety guarantee under arbitrary distribution shift.
5. Do not use target cancer labels for the zero-label experiment's training, threshold selection, tuning, or adaptation.
6. Target labels may be used **after predictions are frozen** to evaluate sensitivity, dismissal rate, missed cancers and confidence intervals.
7. Distinguish strict source-only transfer from unlabeled target adaptation and from target-label calibration.
8. Use patient-level grouping. Do not split left/right breasts, CC/MLO views, repeated exams, crops, or augmentations across partitions.
9. Do not treat CMMD's biopsy-enriched diagnostic cohort as a population-screening cohort.
10. Do not treat VinDr BI-RADS labels as pathology-confirmed cancer outcomes.
11. Do not describe EMBED as unrestricted public data; its access and model-release terms are restrictive.
12. No model has been trained and no results have been generated by our team yet.
13. Q1 publication is an aspiration, not a promised outcome. No acceptance probability should be invented.

---

## 11. Core reference list

### Core papers

1. **TRUST (2026).** *Threshold-Recalibrated Uncertainty-Safe Training for Certified Dismissal in Breast Cancer Screening.* arXiv:2609.00300. https://arxiv.org/abs/2609.00300
2. **Roschewitz et al. (2023).** *Automatic correction of performance drift under acquisition shift in medical image classification.* Nature Communications 14, 6608. https://doi.org/10.1038/s41467-023-42396-y
3. **Garrucho et al. (2022).** *Domain generalization in deep learning based mass detection in mammography: A large-scale multi-center study.* Artificial Intelligence in Medicine 132, 102386. https://doi.org/10.1016/j.artmed.2022.102386
4. **Chen et al. (MICCAI 2025).** *Domain Generalization for Mammogram Classification by Suppressing Domain-Specific Features.* https://doi.org/10.1007/978-3-032-04981-0_17
5. **Yang et al. (2026).** *MammoDG: Generalisable Deep Learning Advances Cross-Domain Multi-Center Breast Cancer Screening.* International Journal of Computer Vision 134. https://doi.org/10.1007/s11263-026-02979-w
6. **Hajishafiezahramini et al. (2026 preprint).** *Dataset-Origin Signatures and Shortcut Learning in Screening Mammography AI: A Cross-Dataset Case Study.* arXiv:2607.15416. https://arxiv.org/abs/2607.15416
7. **Bernstein et al. (2026).** *A novel statistical framework for quantifying risks and benefits of AI automation in screening mammography.* PLOS Digital Health. https://doi.org/10.1371/journal.pdig.0001231
8. **Angelopoulos et al. (ICLR 2024).** *Conformal Risk Control.* https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html
9. **Wang et al. (ICLR 2021).** *Tent: Fully Test-Time Adaptation by Entropy Minimization.* https://arxiv.org/abs/2006.10726
10. **Niu et al. (ICLR 2023).** *Towards Stable Test-Time Adaptation in Dynamic Wild World.* https://arxiv.org/abs/2302.12400

### Core datasets

11. **RSNA dataset descriptor (2026).** *Open-Source Dataset for the RSNA Screening Mammography Cancer Detection Challenge.* Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.250375
12. **NLBS dataset descriptor (2025).** *Full Field Digital Mammography Dataset from a Population Screening Program.* Scientific Data 12, 1479. https://doi.org/10.1038/s41597-025-05866-0
13. **IMDB (2026).** *The Indian Mammography DataBase: A Versioned Open Mammography Resource from a Screening-Naive Indian Population for AI Research.* https://doi.org/10.59275/j.melba.2026-bd61
14. **VinDr-Mammo (2023).** Scientific Data. https://doi.org/10.1038/s41597-023-02100-7
15. **CMMD (2023).** *An Online Mammography Database with Biopsy Confirmed Types.* Scientific Data. https://doi.org/10.1038/s41597-023-02025-1
16. **EMBED (2023).** *The EMory BrEast imaging Dataset (EMBED): A Racially Diverse, Granular Dataset of 3.4 Million Screening and Diagnostic Mammographic Images.* Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.220047

---

## 12. One-sentence summary for future context

We are not yet building a new breast cancer AI model: we are first testing whether TRUST's high-recall mammography dismissal threshold, calibrated on RSNA only, transfers to NLBS with no Canadian cancer labels; then we compare it with UPA using unlabeled NLBS images to find out whether a real, unsolved failure remains.
