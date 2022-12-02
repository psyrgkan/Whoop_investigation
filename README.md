# Whoop_investigation

----
### Project Description


This project explores data from the user’s own wearable health tracking devices. Specifically, it exports data from a WHOOP, a sports’ wearable device focused on performance and recovery of the athlete. The data includes multiple features, such as sleep tracking, Resting Heart Rate (RHR), Heart Rate Variability (HRV) and many more, along with two target variables called Recovery and Strain, which is what is presented to the user after processing of the data. The project also uses data from Apple Health, an iOS app on the user’s iPhone, which has been collecting various data over the years, the most important and consistent of which is Steps, namely how many steps the user performed during the day.

The goal of the project is to investigate the relationships between resting heart rate, hrv, steps and activity during a day, measured by the user’s own wearable device and iPhone over time. Further, the hidden relationship between all of the above variables and Recovery and Strain is going to be examined, trying to produce the best possible model to predict them based on the measurements collected by the devices. An extra step to what the app provides the user as feedback would be to examine behaviors of the user and try to predict Recovery, RHR and HRV of the user for future timesteps based on current measurements.
