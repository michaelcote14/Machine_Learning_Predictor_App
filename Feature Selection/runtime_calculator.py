

def iterator_runtime_predictor(runtimes):
    PredictedTime = 41.2*runtimes + 4.5
    NewTime = PredictedTime

    seconds_in_day = 60 * 60 * 24
    seconds_in_hour = 60 * 60
    seconds_in_minute = 60

    seconds = int(PredictedTime)
    days = seconds // seconds_in_day
    hours = (seconds - (days * seconds_in_day)) // seconds_in_hour
    minutes = (seconds - (days * seconds_in_day) - (hours * seconds_in_hour)) // seconds_in_minute
    print('Runtime Predictor:', days, "days", hours, "hours", minutes, "minutes")

# iterator_runtime_predictor(8)

# total_score = 0
# score = .1
# for o in range(3):
#     for i in range(5):
#         total_score = total_score + score
#         print('inside total score', total_score)
#     print('outside total score', total_score)
#     total_score = 0