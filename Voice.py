import speech_recognition as sr
import pyttsx3
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 

# Change these according to your credentials
fromaddr =  "mishtudeep.@gmail.com"
toaddr   =  "shreedeep.g@mefy.care" 
passwd   =  "Sreeshree123#"

##################################### VOICE RECOGNITION #########################################

def recognize():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    try:
        recognizedAudio =  r.recognize_google(audio)
        print("You said : " + recognizedAudio)
        
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    return recognizedAudio

####################################### VOICE PROMPT ############################################
        
def speak(word):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)    
    engine.setProperty('volume',1.0)  
    voices = engine.getProperty('voices')   
    engine.setProperty('voice', voices[1].id)
    engine.say(word)
    engine.runAndWait()
    engine.stop()

######################################### SEND MAIL ##############################################

def sendMail():
    
    msg = MIMEMultipart()  
    msg['From'] = fromaddr 
    msg['To'] = toaddr 
    msg['Subject'] = "Mail test from Python end."
    body = "Hi, I am from MeFy. I am being tested by my creator." 
    msg.attach(MIMEText(body, 'plain')) 
    filename = "MeFy1.jpg"								  # Provide the file name
    attachment = open("C:\\Users\\SREEDEEP\\Documents\\face\\Mail sender\\MeFy1.jpg", "rb")    # Provide the complete address to the file within " "
    p = MIMEBase('application', 'octet-stream') 
    p.set_payload((attachment).read()) 
    encoders.encode_base64(p) 
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)  
    msg.attach(p) 
    s = smtplib.SMTP('smtp.gmail.com', port=587, timeout=25) 
    s.starttls() 
    s.login(fromaddr, passwd) 
    text = msg.as_string() 
    s.sendmail(fromaddr, toaddr, text) 
    s.quit()

###################################################################################################


#~~~~~~~~~~~ EXECUTION BEGINS FROM HERE ~~~~~~~~~~~#

speak("Welcome, to Me Phi. How may I help you?")
Word = recognize()
speak("you said"+Word)


# if (Word == "please send an email"):
#     speak("Confirm sending email by saying yes or no")
#     print ("Speak now!")
#     Word = recognize()

#     if (Word == "yes"):
#         speak("you said "+Word)
#         speak("sending email")
#         sendMail()
#         speak("email sent successfully")

#     elif (Word == "no"):
#         speak("you said "+Word)
#         speak("cancelled sending email")

# else:
#     speak("Sorry, I have limited abilities now. So, I did not get you.")
#     print(" Format didn't match.\n Couldn't send e-mail.\n Please try again....")
