Loops = 3
PredictedTime = 0.0005607832739*17.28**Loops
NewTime = PredictedTime

seconds_in_day = 60 * 60 * 24
seconds_in_hour = 60 * 60
seconds_in_minute = 60

seconds = int(PredictedTime)
days = seconds // seconds_in_day
hours = (seconds - (days * seconds_in_day)) // seconds_in_hour
minutes = (seconds - (days * seconds_in_day) - (hours * seconds_in_hour)) // seconds_in_minute
print(days, "days", hours, "hours", minutes, "minutes")




