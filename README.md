**Enhancing Hospital-Acquired Infection Surveillance Through Integration of Machine Learning Risk Scoring and CUSUM Sequential Monitoring**

Healthcare-Associated Infections (HAIs) are infections acquired during hospital stay (after 48 hours of hospitalization) [1]. Types of HAIs include catheter-associated urinary tract infections (CAUTI), central venous catheter-associated bloodstream infections (CLABSI), surgical site infections (SSI), and gastrointestinal infections, pneumonia [1]. HAIs cause a burden of disease for patients, increase mortality rates, reduce the quality of treatment and care, prolong treatment time, and increase treatment costs [2]. HAIs overload hospitals and healthcare systems at all levels, reduce hospital quality and reputation, and increase the risk of occupational burnout for healthcare workers [3,4]. Although evidence on the economic burden of HAIs is limited, especially in low- and middle-income countries, available data from the United States and Europe show impacts amounting to billions of dollars. According to the US Centers for Disease Control and Prevention, the total annual direct medical cost of HAIs to hospitals in the US ranges from $35.7 to $45 billion [5], while the annual economic impact in Europe is up to €7 billion [6].

Hospital-acquired infections cause more than 140,000 deaths globally each year, often occurring in healthcare systems of developing countries [7]. European estimates show that more than 4 million patients are affected by approximately 4.5 million hospital-acquired infections (HAIs) each year, resulting in 16 million extended hospital stays, 37,000 deaths and contributing to an additional 110,000 infections. Contributing factors to HAIs are diverse, including pathogen characteristics, host factors, treatment-related aspects, healthcare processes and environmental conditions [8]. While endogenous origin is the primary source of pathogens, human and environmental transmission during healthcare delivery is also a concern.

During the COVID-19 pandemic, hospital-acquired infections (HAIs) became more alarming and a greater concern than ever before. Despite decades of infection prevention and control programs, outbreak detection in most hospitals relied on retrospective review of microbiological culture results, manual surveillance loops, and threshold-based alert systems—which were less sensitive in detecting the gradual and localized increase in infection rates by department. These conventional methods were further hampered by differences in patient demographics: intensive care units and surgical departments consistently recorded higher gross infection rates than general internal medicine departments, not due to poorer infection control, but because of the high-risk nature of the patient population. Any sustainable surveillance system must be able to separate the actual number of infections exceeding the baseline rate adjusted for patient severity.

Two methodological advances offer a path forward. Machine learning (ML) models, trained on routinely collected electronic health record (EHR) data, can generate individual patient-level predicted infection probabilities that accurately account for age, comorbidity burden, invasive device use, immunosuppression, and length of stay. CUSUM (Cumulative Sum) control charts, developed originally for industrial quality control by Page (1954) and subsequently validated in surgical outcome monitoring, can detect sustained shifts in a sequential data stream with far greater power than point-in-time comparisons [10,11]. The conceptual synthesis is elegant: ML provides the risk-adjusted expected rate per patient; CUSUM monitors whether observed outcomes persistently exceed those expectations at the ward level, triggering a prospective alert when the accumulated evidence crosses a predetermined threshold.

Despite the theoretical appeal of this hybrid approach, no prospective multi-site study has evaluated it in real-world HAI surveillance. Published work has largely been restricted to retrospective simulations, single-site pilots, or methodological demonstrations. Critical practical questions remain unanswered, including which ML architecture performs best under real-world class imbalance and data quality constraints, how CUSUM parameters should be calibrated to achieve acceptable false alarm rates across wards with different patient volumes, and what the minimum detectable outbreak size and detection latency are under operational conditions.

**Objectives**
1. Develop and validate an ML ensemble model (Logistic Regression, Random Forest, XGBoost) to generate individual patient HAI risk scores using routinely available EHR variables.
2. Derive data-adaptive CUSUM parameters (k, h) that achieve a target Average Run Length under the null hypothesis (ARL₀ ≥ 500 ward-admissions) while maintaining sensitivity to outbreaks of odds ratio ≥ 2.0.
3. Compare detection latency (patients from outbreak onset to first alarm), false alarm rate, and positive predictive value of ML-CUSUM versus the hospital’s current incident-reporting protocol across five clinical wards over 24 months.
4. Evaluate model calibration stability under realistic covariate shift (seasonal patient mix variation, antibiotic stewardship policy changes) using prospective re-calibration windows

**Methods** 

*Study Design and Setting*
This study employs a prospective, observational, multi-site comparative surveillance design. Three tertiary teaching hospitals (combined capacity: approximately 1,800 beds, approximately 45,000 admissions per year) will participate. Each hospital contributes two to three wards selected to represent a range of acuity levels: medical intensive care, general surgery, oncology, general medicine, and orthopaedics. Data collection proceeds in two phases: a retrospective model training phase (months 1–12, using three years of historical EHR data) and a prospective surveillance phase (months 13–24).

*Data Sources and Variables*
Predictor variables will be extracted from the hospital EHR at admission and updated daily. Core predictors include patient age, sex, body mass index, length of stay, Charlson Comorbidity Index score, admission diagnosis ICD-10 code group, ICU admission status, surgical procedure (yes/no, type), presence and duration of central venous catheter, urinary catheter, and mechanical ventilation, immunosuppressive medication use, broad-spectrum antibiotic exposure (defined daily doses), and prior HAI history within 12 months.

Outcomes will be defined per CDC/NHSN criteria: HAI is confirmed if a qualifying organism is isolated from a culture taken more than 48 hours after admission, meeting site-specific case definitions for CAUTI, CLABSI, SSI, CDI, or hospital-acquired pneumonia. Daily ward-level denominator data (device-days, patient-days) will be recorded for rate calculation.

*Machine learning Model Development Pipeline*
Three candidate models will be developed: regularised logistic regression with clinically motivated interaction terms (central line duration × immunosuppression; ICU admission × comorbidity score); a balanced Random Forest (sampsize set to equalise class proportions, mtry tuned by out-of-bag error); and XGBoost with scale_pos_weight = n_negative / n_positive to address the approximately 5–10% HAI prevalence. All models will be trained on the retrospective cohort using a temporal cross-validation scheme (training on years 1–2, validation on year 3) to prevent leakage of temporal trends. Model calibration will be assessed using isotonic regression via five-fold cross-validation. The final deployed score will be a weighted ensemble, with weights proportional to cross-validated AUC, updated quarterly during the prospective phase.

*CUSUM Implementation*
The CUSUM statistic will be computed at the ward level using the upward one-sided algorithm: Sₙ = max(0, Sₙ₋₁ + (Oᵢ − Eᵢ) − k), where Oᵢ is the observed binary infection outcome and Eᵢ is the ML ensemble score for patient i. An alert fires when Sₙ > h. The reference value k will be set as half the minimum detectable residual shift (δ/2), estimated empirically from historical outbreak records. The decision threshold h will be derived via parametric bootstrap simulation of 10,000 null sequences to achieve the target ARL₀. Ward-specific parameters will account for differences in patient volume and baseline risk. The CUSUM statistic will be reset to zero following each alert and after a formal clinical investigation is initiated.

**Experiment Results**
In order to have the data to learn and practice the combination of advanced techniques, I generated and simuluated the synthetic data.
There are 8 main steps to conduct this method. 

Section 0 — Environment Setup
Sets set.seed(2024) to make all random draws reproducible, then installs and loads 7 packages in a loop. The character.only = TRUE argument in library() is needed because the package name is stored in a variable, not typed as a bare name.
Section 1 — Data Simulation
Generates 3,000 synthetic patients using a logistic model with 10 clinical risk factors drawn from the published HAI literature. The outbreak is injected as log(OR) added to the log-odds of Ward-3 patients starting at index 2,200 — this correctly multiplies the odds by exactly 3.0 without distorting the rest of the model.
Section 2 — Explotary Data Analysis
Computes HAI rates per ward and crude odds ratios per risk factor. This step is essential before modelling — it verifies the simulation behaves as intended.
Section 3 — Machine learning Models
Trains Logistic Regression and Random Forest on patients 1–1,500. Critically, ward is deliberately excluded as a predictor — otherwise the model would absorb the outbreak signal instead of leaving it for CUSUM to detect. Platt scaling corrects probability overconfidence from the LR model.
Section 4 — Model Evaluation
Measures discrimination (AUC), overall accuracy (Brier Score), and calibration (Hosmer-Lemeshow test + decile calibration table). Calibration is especially important here because CUSUM uses observed - predicted as its input signal.
Section 5 — CUSUM Function
The run_cusum() function implements the Page (1954) formula with dual-direction monitoring (upward for outbreak detection, downward for quality improvement), configurable k/h parameters, and a reset-after-alert policy.
Section 6 — 7 Publication-Quality Plots
Risk distribution, ROC curves, calibration plot, per-ward CUSUM panels, Ward-3 residual stream, feature importance, and sensitivity heatmaps — all saved as PNG files.
Section 7 — Sensitivity Analysis
A full grid over k ∈ {0.2–0.9} × h ∈ {2.0–6.0} showing detection delay and false alarm count for every combination, saved as two heatmaps.
Section 8 — Report
Prints a boxed summary table with all key metrics and five numbered research insights explaining what the simulation teaches about the method.
















































**Reference**

[1]  Nguyễn Kiến Mậu, Tăng Kim Hồng. (2018). Tỷ suất mới mắc nhiễm khuẩn bệnh viện tại khoa Sơ sinh bệnh viện Nhi Đồng. Nghiên cứu Y học, 10 (3), tr. 67 – 71. 
[2]  Nhiễm khuẩn bệnh viện làm tăng tỷ lệ mắc bệnh, nguy cơ tử vong, kéo dài ngày nằm viện [Truy cập: https://bvquan5.medinet.gov.vn/chuyen-muc/nhiem-khuan-benh-vien-lam-tang-ty-le-mac-benh-nguy-co-tu-vong-keo-dai-ngay-nam-cmobile14478-224747.aspx]
[3] Thủ tướng: Chính phủ quyết tâm cải thiện tình trạng quá tải bệnh viện, chất lượng dịch vụ y tế gắn với cơ chế tiền lương. [Truy cập: https://medinet.gov.vn/tin-tuc-su-kien/thu-tuong-chinh-phu-quyet-tam-cai-thien-tinh-trang-qua-tai-benh-vien-chat-luong-cmobile1780-37838.aspx]
[4] Quá tải tại một số bệnh viện tuyến Trung ương [Truy cập: https://nhandan.vn/qua-tai-tai-mot-so-benh-vien-tuyen-trung-uong-post707866.html]
[5] Scott R II. The direct medical costs of healthcare-associated infections in U.S. hospitals and the benefits of prevention. Atlanta (GA): Centers for Disease Control and Prevention; 2009. [18 October 2016]. https://www.cdc.gov/HAI/pdfs/hai/Scott_CostPaper.pdf
[6] Report on the state of communicable diseases in the EU and EEA/EFTA countries. Stockholm: European Centre for Disease Prevention and Control; 2008. [18 October 2016]. Annual epidemiological report on communicable diseases in Europe 2008. http://ecdc.europa.eu/en/publications/Publications/0812_SUR_Annual_Epidemiological_Report_2008.pdf.
[7] World Health Organization. (2011). Report on the burden of endemic health care-associated infection worldwide.
[8] Allegranzi, B., Nejad, S. B., Combescure, C., Graafmans, W., Attar, H., Donaldson, L., & Pittet, D. (2011). Burden of endemic health-care-associated infection in developing countries: systematic review and meta-analysis. The Lancet, 377(9761), 228-241.
[9] Guideline, I. P. (2007). Guideline for Isolation Precautions: Preventing Transmission of Infectious Agents in Healthcare Settings (2007).
[10] Poloniecki JD, et al. (1998). Cumulative risk adjusted mortality chart for detecting changes in death rate. BMJ, 316(7146), 1697–1700.
[11]  Sherlaw-Johnson C. (2005). A method for detecting runs of good and bad clinical outcomes on variable life-adjusted display (VLAD) charts. Health Care Management Science, 8(1), 61–68.




