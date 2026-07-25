"""Bag loading for the id_analyser scripts.

ROS2 port (2026-07-23): the original ForzaETH version used bagpy, which only
reads ROS1 .bag files. This version uses `rosbags` (pip install rosbags),
whose AnyReader transparently reads ROS1 .bag, ROS2 sqlite3 (.db3) and ROS2
mcap bags - so old and new recordings both work.

Interface is unchanged for the analyser scripts:
    load_bags(directory, field_dict) -> pandas.DataFrame
      field_dict: {topic: [dotted.field, ...]}  e.g.
        {"/car_state/odom": ["twist.twist.linear.x", "twist.twist.linear.y"],
         "/vesc/commands/servo/position": ["data"]}
      Columns in the result:
        - requested fields under their dotted names
        - std_msgs Float64 `data` renamed to "<topic>.data"
        - "header.stamp.nsecs": int64 epoch nanoseconds used for alignment
      Topics are time-aligned with pandas merge_asof (nearest match) onto the
      topic with the fewest samples, mirroring the original behaviour.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

TIME_COL = "header.stamp.nsecs"  # kept from the ROS1 loader for compatibility


def _get_dotted(msg, dotted):
    """Resolve 'twist.twist.linear.x' style paths on a deserialized message."""
    obj = msg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _stamp_ns(msg, fallback_ns):
    """Header stamp in epoch ns; bag receive time when the msg is unstamped."""
    header = getattr(msg, "header", None)
    if header is not None:
        stamp = header.stamp
        # ROS2: sec/nanosec, ROS1 (via rosbags): sec/nanosec as well
        ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if ns > 0:
            return ns
    return int(fallback_ns)


def _topic_df(reader, topic, fields):
    """One topic -> DataFrame[fields..., TIME_COL], time-sorted."""
    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        available = sorted({c.topic for c in reader.connections})
        raise ValueError(
            f"topic {topic} not in bag; available topics: {available}")
    rows = []
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        row = {f: _get_dotted(msg, f) for f in fields}
        row[TIME_COL] = _stamp_ns(msg, timestamp)
        rows.append(row)
    if not rows:
        raise ValueError(f"topic {topic} exists but has no messages")
    df = pd.DataFrame(rows).sort_values(TIME_COL, ignore_index=True)

    # handle commands via std_msgs Float64 (bare `data` field)
    df.rename(columns={"data": topic + ".data"}, inplace=True)

    # handle default ackermann drive commands: drop all-zero keepalives
    if "drive.speed" in df.columns and "drive.acceleration" in df.columns:
        df = df[~((df["drive.speed"] == 0.0)
                  & (df["drive.acceleration"] == 0.0))].reset_index(drop=True)
    return df


def get_bag_df(bagpath, field_dict):
    """Load one bag (ROS1 .bag / ROS2 dir / .db3 / .mcap) into one merged df."""
    print(f"Loading Bag {bagpath}...")
    with AnyReader([Path(bagpath)]) as reader:
        topic_dfs = []
        for topic, fields in field_dict.items():
            print("___ IMPORTING TOPIC " + topic)
            topic_dfs.append(_topic_df(reader, topic, fields))

    # crop every topic to the overlapping time window
    start = max(df[TIME_COL].iloc[0] for df in topic_dfs)
    end = min(df[TIME_COL].iloc[-1] for df in topic_dfs)
    topic_dfs = [df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end)]
                 for df in topic_dfs]
    if any(df.empty for df in topic_dfs):
        raise ValueError(
            f"{bagpath}: topics do not overlap in time - check the recording")

    # merge onto the sparsest topic (nearest-timestamp match), like the original
    topic_dfs.sort(key=len)
    bag_df = topic_dfs[0]
    for df in topic_dfs[1:]:
        bag_df = pd.merge_asof(bag_df, df, on=TIME_COL, direction="nearest")
    bag_df.dropna(inplace=True)
    bag_df = bag_df.astype({TIME_COL: "int64"})
    return bag_df


def _is_bag(path):
    """ROS1 .bag file, ROS2 bag directory (metadata.yaml), .db3 or .mcap file."""
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, "metadata.yaml"))
    return os.path.splitext(path)[1] in (".bag", ".db3", ".mcap")


def load_bags(directory, field_dict):
    """Load and concatenate every bag found directly inside `directory`."""
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist")
    entries = sorted(os.listdir(directory))
    if not entries:
        raise ValueError(f"Directory {directory} is empty")
    dataframe_list = []
    for filename in entries:
        f = os.path.join(directory, filename)
        if _is_bag(f):
            dataframe_list.append(get_bag_df(f, field_dict))
    if not dataframe_list:
        raise ValueError(
            f"No bags (ROS1 .bag / ROS2 dir / .db3 / .mcap) found in {directory}")
    bags_df = pd.concat(dataframe_list, ignore_index=True)
    print("Total number of datapoints: " + str(bags_df.shape[0]))
    return bags_df
