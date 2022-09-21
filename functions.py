def runtime_predictor(runtimes):
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



#adds text to a file
def text_file_appender(filename, text_to_add):
    with open (filename, 'a') as file:
        file.write(text_to_add)
        file.close()



if __name__ == "__main__":
    runtime_predictor(3)
    text_file_appender('test_data', ['blue', 'yellow'])

text_data = ['blue', 'yellow']
