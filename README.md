# ECG Classification
## Introduction to ECG:
### ECG Signal:
An electrocardiogram (ECG) is a test that measures the electrical activity of the heartbeat. With each beat, an electrical impulse (or “wave”) travels through the heart. This wave causes the muscle to squeeze and pump blood from the heart. A normal heartbeat on ECG will show the timing of the top and lower chambers.

![5  cm](https://ars.els-cdn.com/content/image/1-s2.0-S0306987719312381-gr1.jpg)

- P-Wave: Depolarization of the atrial myocardium.
- PR interval: It is the wave goes over the atrium and through the AV node and ends just before it activates the ventricles to depolarize.
- PR segment: It is the depolarization of the AV node.
- QRS Complex: Depolarization of the ventricular myocardium.
- ST segment: During the ST segment, all the ventricular myocardium is depolarized. All have positive charges. So, there is nothing potential difference to be recorded by the voltmeter (ECG machine). So, you have a flat line.
- QT interval: It is Important because it captures the beginning of ventricular depolarization through the plateau phase to the ventricular repolarization. It covers the entire ventricular activity. During this time, the action potential was generated and terminated in the ventricular tissue.
- T-wave: Repolarization of the ventricular myocardium. The right and left atria or upper chambers make the first wave called a “P wave" — following a flat line when the electrical impulse goes to the bottom chambers. The right and left bottom chambers or ventricles make the next wave called a “QRS complex". The final wave or “T wave” represents electrical recovery or return to a resting state for the ventricles.

### Arrhythmia:
Arrhythmia is a Heart rhythm problem (heart arrhythmias), which occurs when the electrical impulses that coordinate your heartbeats don't work properly, causing your heart to beat too fast, too slow or irregularly. Thus, any irregularity in heart functions leads to an change in the ECG signal and it can be differentiated from normal ECG signal.

## Objective:
The objective of this project is to classify ECG Signal to one of the following classes:
- 0: Normal Heart beat
- 1: Unknown Heart beat
- 2: Ventricular Ectopic beat
- 3: Superventricular Ectopic beat
- 4: Fusion beat

## Datasets:
The datasets used in this project were downloaded from [Kaggle](https://www.kaggle.com/code/freddycoder/heartbeat-categorization/data). In the link, there are 4 files. Only mitbih_train.csv and mitbih_test.csv are used.

## Files:
This Repository contains 3 files:
- Code.py: this is the code in a .py file.
- ECG_Classification.ipynb: This is the code in a Jupyter Notebook.
- README.md: It is an introduction to the ECG signals and a guide to the repository.
