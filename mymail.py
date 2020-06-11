import smtplib
from email.mime.text import MIMEText


def mail(msg):
    msg_from = '14435635@qq.com'  # 发送方邮箱地址。
    password = 'fbtzyfqsxrcybijb'  # 发送方QQ邮箱授权码，不是QQ邮箱密码。
    msg_to = '14435635@qq.com'  # 收件人邮箱地址。

    subject = "train model log"  # 主题。
    content = msg  # 邮件正文内容。
    msg = MIMEText(content, 'plain', 'utf-8')

    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to
    try:
        client = smtplib.SMTP_SSL('smtp.qq.com', smtplib.SMTP_SSL_PORT)
        print("连接到邮件服务器成功")

        client.login(msg_from, password)
        print("登录成功")

        client.sendmail(msg_from, msg_to, msg.as_string())
        print("发送成功")
    except smtplib.SMTPException as e:
        print("发送邮件异常")
    finally:
        client.quit()
