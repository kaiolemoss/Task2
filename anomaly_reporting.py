import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, message, recipient_email):
    sender_email = "your-email@gmail.com" #test
    sender_password = "password"

    # Configure the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Add message body
    msg.attach(MIMEText(message, 'plain'))

    # Configure the SMTP server and send the email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    text = msg.as_string()
    server.sendmail(sender_email, recipient_email, text)
    server.quit()

def report_anomaly(anomalies):
    if anomalies:
        print("Anomaly detected:", anomalies)
        subject = "Anomaly Detected Alert"
        message = f"Anomaly detected in the system: {anomalies}"
        recipient_email = "email-destination@example.com" #test
        send_email(subject, message, recipient_email)
