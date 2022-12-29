
def time_formatter(input_time_in_seconds):
    input_time_in_seconds = float(input_time_in_seconds)
    from datetime import timedelta
    # create timedelta and convert it into string
    td_str = str(timedelta(seconds=input_time_in_seconds))

    # split string into individual component
    x = td_str.split(':')
    time_list = [str(x[0]), 'Hours', str(x[1]), 'Minutes', str(x[2]), 'Seconds']
    time_string = ' '.join(time_list)
    return time_string


#adds text to a file
def text_file_appender(filename, text_to_add):
    with open (filename, 'a') as file:
        file.write(text_to_add)
        file.close()

# Reads up to whatever line you specify
def text_file_reader(filename, letter_index1, letter_index2):
    with open(filename, 'r') as file:
        all_letters = file.read()
        return (all_letters[letter_index1:letter_index2])
        file.close()
        pass

# reads specific lines of a file, just add another readline statement
# to get the next line
# USE THIS FOR EVERTHING!!
def line_checker(filename, line_to_read, letters_to_read='none'):
    with open (filename, 'r') as file:
        print(file.readlines()[1][0:5])



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
    line_checker('test_data', 5)
    time_formatter(9337.2926)


