"""python3 -m carton_picker [--start-slot N] [--step]"""
import argparse

import rclpy

from .moveit_client import MoveItClient
from .workflow import Depalletizer, Pauses


def main():
    ap = argparse.ArgumentParser(description='Depalletize cartons from the source to the destination pallet.')
    ap.add_argument('--start-slot', type=int, default=0,
                    help='number of cartons already on the destination pallet (default 0)')
    ap.add_argument('--step', action='store_true', help='wait for Enter before each stage')
    args = ap.parse_args()

    rclpy.init()
    client = MoveItClient()
    try:
        Depalletizer(client, Pauses(client.get_logger(), args.step)).run(args.start_slot)
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
