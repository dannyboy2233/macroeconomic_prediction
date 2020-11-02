"""
Establishes monthly LaTeX reporting methodology.

Original author: Daniel Cohen, Summer 2019 Competitive Intelligence Analyst Intern
"""


from colorsys import hsv_to_rgb  # standard libraries
import subprocess
import datetime as dt
from contextlib import contextmanager
import sys, os
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate

import pandas as pd  # third-party libraries

from Utils import change_dir, add_months, get_credentials  # package libraries


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_parameters():
    """
    Obtains parameters required for use in export_report; also updates curr_preds.csv and prev_preds.csv.
    :return: list: [curr_preds, prev_preds, next_three_months]
    """

    change_dir("Table exports")
    curr_preds_tbl = pd.read_csv("curr_preds.csv")
    curr_preds = curr_preds_tbl['preds'].values  # [0.8, 1.2, 1.8]
    try:
        prev_preds_tbl = pd.read_csv('prev_preds.csv')  # [0.0, 0.0, 0.0]
        prev_preds = prev_preds_tbl['preds'].values
    except OSError:  # if prev_preds table doesn't exist, there are no previous predictions, and there is no change
        prev_preds = curr_preds

    next_three_months = [add_months(dt.datetime.now(), i).strftime('%B  %Y') for i in range(1, 4)]
    # next_three_months = ['August 2019', 'September 2019', 'October 2019']

    curr_preds_tbl.to_csv("prev_preds.csv", index=False)  # export current predictions as previous predictions

    return [curr_preds, prev_preds, next_three_months]


def export_report(curr_preds, prev_preds, next_three_months, sources):
    """
    Takes predictions/sources table and produces PDF of desired report.
    :param curr_preds: list containing current one-month, two-month, three-month predictions
    :param prev_preds: list containing previous one-month, two-month, three-month predictions;
           to clarify, the element at prev_preds[1] corresponds to the same month as curr_preds[0]
    :param next_three_months: list of names of next three months ['August 2019', 'September 2019', 'October 2019'] e.g.
    :param sources: DataFrame from data warehouse containing key, source for each independent variable
    :return: Nothing, but exports report of desired PDF based on LaTeX string created with new data.
    """

    sources = sources.fillna("")
    hsv_red = 110  # hsv_green = 0

    max_val = 30  # probability cutoff for maximum "danger"
    min_val = 0
    num_dec = 2  # number of decimal places

    colors = []

    deltas = []
    for i in range(3):
        curr_pred = curr_preds[i]
        try:
            curr_delta = curr_pred - prev_preds[i + 1]
        except IndexError:
            curr_delta = 0
        deltas.append(curr_delta)
        total = curr_pred + curr_delta
        if total > max_val:
            total = max_val
        elif total < min_val:
            total = min_val
        hsv_color_decimal = (hsv_red * (1 - (total / max_val))) / 359
        rgb_color = hsv_to_rgb(hsv_color_decimal, 0.5, 1)
        colors.append(map(lambda x: x * 255, rgb_color))

    change_dir('LaTeX')

    color_strs = [r"\definecolor{fillcolor" + str(i) + r"}{RGB}{" + ",".join(map(str, colors[i])) + "}" for i in range(3)]

    rep = {"COLOR1": color_strs[0],
           "COLOR2": color_strs[1],
           "COLOR3": color_strs[2],
           "MONTH1": next_three_months[0],
           "MONTH2": next_three_months[1],
           "MONTH3": next_three_months[2],
           "CURRPRED1": str(round(curr_preds[0], num_dec)),
           "CURRPRED2": str(round(curr_preds[1], num_dec)),
           "CURRPRED3": str(round(curr_preds[2], num_dec)),
           "DELTA1": str(round(deltas[0], num_dec)) if deltas[0] >= 0 else str(round(deltas[0] * -1, num_dec)),
           "DELTA2": str(round(deltas[1], num_dec)) if deltas[1] >= 0 else str(round(deltas[1] * -1, num_dec)),
           "DELTA3": str(round(deltas[2], num_dec)) if deltas[2] >= 0 else str(round(deltas[2] * -1, num_dec)),
           "ARROW1": "$\\downarrow$" if deltas[0] < 0 else "$\\uparrow$",
           "ARROW2": "$\\downarrow$" if deltas[1] < 0 else "$\\uparrow$",
           "ARROW3": "$\\downarrow$" if deltas[2] < 0 else "$\\uparrow$",
           "SRCS": "\n".join([sources['key'].iloc[i] + " & " + sources['source'].iloc[i].replace("_", "\\_") + "\\\\" for i in range(len(sources))])}

    def replace_all(text, dic):
        """
        Function to replace multiple substrings of string.
        :param text: text to replace
        :param dic: dict that maps str to replace to desired str
        :return: modified text
        """
        for j, k in dic.items():
            text = text.replace(j, k)
        return text

    texdoc = []

    with open('report_python_import.tex') as fin:
        for line in fin:
            texdoc.append(replace_all(line, rep))

    subprocess.run(["rm", "report_python_export.tex"])
    subprocess.run(["touch", "report_python_export.tex"])
    with open('report_python_export.tex', 'w') as fout:
        for i in range(len(texdoc)):
            fout.write(texdoc[i])

    with suppress_stdout():  # LaTeX compiling produces too much console text --> silence it! muahahah
        subprocess.run(['pdflatex', 'report_python_export.tex'])


def send_email(recipients):
    """
    Sends email containing report_python_export.pdf to recipients
    :param recipients: list of email addresses to receive the report.
    :return: Nothing, but sends the email.
    """

    print("Sending email...")

    # set up the SMTP server with throwaway email
    email_creds = get_credentials('email_creds.txt')  # NOTE: email address is not secure. use only for these reports
    s = smtplib.SMTP(host='smtp.gmail.com', port=587)
    s.starttls()
    s.login(email_creds['user'], email_creds['pw'])

    send_from = email_creds['user']
    subject = 'Macroeconomic Forecast: {}'.format(dt.datetime.strftime(dt.datetime.today(), "%B %d, %Y"))
    text = (
        "Hi! \n"
        "Attached is this month's macroeconomic forecast, "
        "designed by Danny Cohen (Corporate Intelligence Analyst Intern, Summer 2019). \n"
        "I hope this finds you well."
    )

    change_dir('LaTeX')
    files = ['report_python_export.pdf']

    parts = []

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        parts.append(part)

    for r in recipients:
        msg = MIMEMultipart()
        msg['From'] = 'Report Manager'
        msg['To'] = r
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subject
        msg.attach(MIMEText(text))

        [msg.attach(part) for part in parts]

        s.sendmail(send_from, r, msg.as_string())
    s.close()
