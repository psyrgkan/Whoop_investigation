create database if not exists whoop;

use whoop;

select truncate(avg(`Energy burned (cal)`), 0) as `Average Calories`,
		truncate(avg(`Energy burned (cal)`/`Duration (min)`), 2) as `Average Calories per Minute`, 
        truncate(avg(`Average HR (bpm)`), 0) as `Average HR (bpm)`, `Activity name` as Sport
from workout
group by `Activity name`
order by `Average Calories`
desc
limit 5;

select min(`Resting heart rate (bpm)`) as `Min RHR`, max(`Resting heart rate (bpm)`) as `Max RHR`,
	min(`Heart rate variability (ms)`) as `Min HRV`, max(`Heart rate variability (ms)`) as `Max HRV`
from physio;

select count( `Activity name`) as count,  `Activity name` as Sport
from workout
group by Sport
order by count desc
limit 4;

select `Activity name` as Sport,
		truncate(avg(`HR Zone 1 %`),1) as `Avg Zone 1`,
		truncate(avg(`HR Zone 2 %`),1) as `Avg Zone 2`,
		truncate(avg(`HR Zone 3 %`),1) as `Avg Zone 3`,
		truncate(avg(`HR Zone 4 %`),1) as `Avg Zone 4`,
		truncate(avg(`HR Zone 5 %`),1) as `Avg Zone 5`
from workout
group by Sport
order by count(Sport) desc
limit 4;

select truncate(avg(`Asleep duration (min)`),1) as `Average Sleep`, 
		truncate(avg(`Sleep performance %`),1) as `Average Sleep Performance`,
        truncate(avg(`Hour wake`),1) as `Average wake up`,
        truncate(avg(`Sleep need (min)`),1) as `Average Sleep need (min)`,
        truncate(avg(`Sleep debt (min)`),1) as `Average Sleep debt (min)`
from sleep
GROUP BY if(`Date` < "2022-10-10" , 0, 1);

select month(creationDate), avg(value) as `Average steps per weekday`
from steps
group by MONTH(creationDate)
order by MONTH(creationDate);
