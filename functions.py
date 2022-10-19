def feature_runtime_predictor(runtimes):
    PredictedTime = 94519.87*runtimes + 4.785196793

    seconds_in_day = 60 * 60 * 24
    seconds_in_hour = 60 * 60
    seconds_in_minute = 60

    seconds = int(PredictedTime)
    days = seconds // seconds_in_day
    hours = (seconds - (days * seconds_in_day)) // seconds_in_hour
    minutes = (seconds - (days * seconds_in_day) - (hours * seconds_in_hour)) // seconds_in_minute
    print('Runtime Predictor:', days, "days", hours, "hours", minutes, "minutes")

def trainer_runtime_predictor(predicted_time):
    seconds_in_day = 60 * 60 * 24
    seconds_in_hour = 60 * 60
    seconds_in_minute = 60

    seconds = int(predicted_time)
    days = seconds // seconds_in_day
    hours = (seconds - (days * seconds_in_day)) // seconds_in_hour
    minutes = (seconds - (days * seconds_in_day) - (hours * seconds_in_hour)) // seconds_in_minute
    print('Time:', days, "days", hours, "hours", minutes, "minutes")

def seconds_formatter(elapsed_time):

    seconds_in_day = 60 * 60 * 24
    seconds_in_hour = 60 * 60
    seconds_in_minute = 60

    seconds = int(elapsed_time)
    days = seconds // seconds_in_day
    hours = (seconds - (days * seconds_in_day)) // seconds_in_hour
    minutes = (seconds - (days * seconds_in_day) - (hours * seconds_in_hour)) // seconds_in_minute
    print('Time:', days, "days", hours, "hours", minutes, "minutes")


#adds text to a file
def text_file_appender(filename, text_to_add):
    with open (filename, 'a') as file:
        file.write(text_to_add)
        file.close()

# Reads up to whatever line you specify
def text_file_reader(filename, letter_index1, letter_index2):
    with open(filename, 'r') as file:
        all_letters = file.read()
        print(all_letters[letter_index1:letter_index2]) #do this to select which letters you want to read in
        return (all_letters[letter_index1:letter_index2])
        file.close()
        pass

# reads specific lines of a file, just add another readline statement
# to get the next line
def line_checker(filename, line_length):
    with open (filename, 'r') as file:
        print(file.readline())
        line_to_check = (len(file.readline()))
        print(line_to_check)
        if line_to_check > 34:

            print('worked')

        file.close()


def email_or_text_alert(subject, body, to):
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to

    user = 'michaelcote14@gmail.com'
    msg['from'] = user
    password = 'bbsfeulvmgjvywhg'

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)
    server.quit()




if __name__ == "__main__":

    feature_runtime_predictor(3)
    line_checker('test_data', 5)
    seconds_formatter(9337.2926)



