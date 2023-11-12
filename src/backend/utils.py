# -*- coding: utf-8 -*-
import ast
import functools
import logging
import os
import re
import time
from datetime import date, datetime, timedelta

from dateutil.parser import parse as date_parser


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # Start measuring the time
        value = func(*args, **kwargs)  # Call the actual function
        end_time = time.perf_counter()  # Stop measuring the time
        run_time = end_time - start_time  # Calculate runtime

        logging.info(f"Finished {func.__name__!r} in {run_time:.4f} seconds.")
        return value

    return wrapper_timer


# Helper Functions
def truncate_text(text, max_chars=1028):
    return text[:max_chars]


def preprocess_comment(comment):
    if comment.startswith("[") and comment.endswith("]"):
        try:
            comment_list = ast.literal_eval(comment)
        except (SyntaxError, ValueError):
            comment_list = [comment]  # fallback behavior
    else:
        comment_list = [comment]

    comment = " ".join(comment_list)
    comment = re.sub(r"#", "", comment)
    return truncate_text(comment, 500)


def preprocess_data(data):
    # Convert 'True' to 1 and any other value (including NaN) to 0
    data["exploitability"] = data["exploitability"].apply(
        lambda x: 1 if x is True else 0
    )
    return data


def _get_date_from_str(dts, default):
    if isinstance(dts, int):
        dts = str(dts)
    if dts is not None:
        if dts == "today":
            dt = date.today()
        elif dts == "yesterday":
            dt = date.today() - timedelta(1)
        elif dts == "two_days_ago":
            dt = date.today() - timedelta(2)
        elif dts == "three_days_ago":
            dt = date.today() - timedelta(3)
        elif dts == "four_days_ago":
            dt = date.today() - timedelta(4)
        elif dts == "five_days_ago":
            dt = date.today() - timedelta(5)
        else:
            dt = date_parser(dts).date()
    else:
        dt = date_parser(default).date()
    return dt


def parse_start_date(startdate_str):
    return _get_date_from_str(startdate_str, "1950-01-01")
    # startdate = startdate.strftime("%Y%m%d")


def parse_end_date(enddate_str):
    return _get_date_from_str(enddate_str, "2199-01-01")
    # startdate = startdate.strftime("%Y%m%d")


def parse_end_date_inclusive(enddate_str):
    return _get_date_from_str(enddate_str, "2199-01-01") + timedelta(1)
    # startdate = startdate.strftime("%Y%m%d")


def get_date_file_modified(file_path):
    return datetime.fromtimestamp(os.path.getmtime(file_path))  # .replace(tzinfo=None)


def is_valid_date(daterange, recorddate):
    if daterange[0] is not None and recorddate < daterange[0]:
        return False
    if daterange[1] is not None and recorddate > daterange[1]:
        return False
    return True
