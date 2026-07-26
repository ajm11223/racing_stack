import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from helpers.load_model import get_dotdict
from helpers.vehicle_dynamics import vehicle_dynamics

# ROS2 port (2026-07-26): upstream pulled the car/tire configs from ForzaETH's
# `pbl_config` package, which we don't vendor. Both configs are plain parameter
# bags, so they are built here from the id_analyser model file that
# analyse_tires.py already writes (models/<name>/<name>_pacejka.txt).


@dataclass
class CarConfig:
    m: float
    lf: float
    lr: float
    h_cg: float
    Iz: float


@dataclass
class PacejkaTireConfig:
    friction_coeff: float
    Bf: float
    Cf: float
    Df: float
    Ef: float
    Br: float
    Cr: float
    Dr: float
    Er: float


def load_configs(model_name, mu=None):
    """model_name e.g. 'UNICORN2-0726_pacejka'; mu overrides the floor grip."""
    m = get_dotdict(model_name)
    car = CarConfig(m=m.m, lf=m.l_f, lr=m.l_r, h_cg=m.h_cg, Iz=m.I_z)
    Bf, Cf, Df, Ef = m.C_Pf
    Br, Cr, Dr, Er = m.C_Pr
    tire = PacejkaTireConfig(
        friction_coeff=m.mu if mu is None else mu,
        Bf=Bf, Cf=Cf, Df=Df, Ef=Ef, Br=Br, Cr=Cr, Dr=Dr, Er=Er)
    return car, tire

# Simulation parameters
SIMULATION_DURATION = 2.0  # seconds
SIMULATION_DT = 0.01  # seconds
PLOT_LOOKUP = True

# Lookup parameters
START_STEER = 0.0  # rad
STEER_FINE_END = 0.1  # rad
FINE_STEP_SIZE = 0.0033  # rad
END_STEER = 0.4  # rad
COARSE_STEP_SIZE = 0.01  # rad
START_VEL = 0.5  # m/s
END_VEL = 7.0  # m/s
VEL_STEP_SIZE = 0.1  # m/s

class Simulator:
    def __init__(self, model_name: str, mu: float = None):
        self.car_config, self.pacejka_config = load_configs(model_name, mu)
        self.sol = None

    def func_ST(self, x, t, u):
        """Wrapper to the vehicle dynamics function to be used with scipy.integrate.odeint"""
        f = vehicle_dynamics(x, u, self.car_config, self.pacejka_config)
        return f

    def run_simulation(self, initialState, u,
                       duration=SIMULATION_DURATION, dt=SIMULATION_DT):
        t = np.arange(0, duration, dt)
        self.sol = odeint(self.func_ST, initialState, t, args=(u,))
        return self.sol


class LookupGenerator:
    def __init__(self, model_name: str, table_name: str, mu: float = None,
                 update_latest: bool = True):
        self.sim = Simulator(model_name, mu)
        self.model_name = model_name
        self.table_name = table_name
        self.update_latest = update_latest
        self.lookup_table = None

    def run_generator(self):
        self.generate_lookup()
        self.find_upper_limits()
        if PLOT_LOOKUP:
            self.plot_lookup()
        self.save_lookup()

    def generate_lookup(self):
        fine_steers = np.linspace(START_STEER, STEER_FINE_END, int(
            (STEER_FINE_END - START_STEER) / FINE_STEP_SIZE), endpoint=False)
        coarse_steers = np.linspace(STEER_FINE_END, END_STEER, int((END_STEER - STEER_FINE_END) / COARSE_STEP_SIZE))
        steers = np.concatenate((fine_steers, coarse_steers))
        vels = np.linspace(START_VEL, END_VEL, int((END_VEL - START_VEL) / VEL_STEP_SIZE))
        n_steps_steer = len(steers)
        n_steps_vel = len(vels)

        self.lookup_table = np.empty([n_steps_steer + 1, n_steps_vel + 1])
        self.lookup_table[0, 1:] = vels
        self.lookup_table[1:, 0] = steers

        for steer_idx, steer in enumerate(steers):
            for vel_idx, vel in enumerate(vels):
                initialState = [0, 0, 0, vel, 0, 0]
                u = [steer, 0]
                sol = self.sim.run_simulation(initialState, u)

                # check if sol[5] is does not change anymore - steady state reached
                if np.allclose(sol[-11:-1, 5], sol[-15:-5, 5], rtol=1e-3):
                    # record the final lateral acceleration
                    a_lat = sol[-1, 5] * vel
                    self.lookup_table[steer_idx + 1, vel_idx + 1] = a_lat
                else:
                    # No steady state solution found
                    # No need to continue with this steering angle for higher velocities
                    self.lookup_table[steer_idx + 1, vel_idx + 1:] = None
                    break

    # we don't want multiple steering angle and velocity combinations to have the same lateral acceleration
    # sol: only take the lower combinations of steering angle and velocity
    # in parallel find the upper limit for the achievable lateral acceleration
    def find_upper_limits(self):
        # find the first maximum for a velocity and then only keep values on the lower end, set everything above to nan
        for vel_idx in range(1, self.lookup_table.shape[1]):
            vel = self.lookup_table[0, vel_idx]
            a_lats = self.lookup_table[1:, vel_idx]
            a_lats = a_lats[~np.isnan(a_lats)]  # drop nans at the end
            d_a_lats = np.diff(a_lats)
            # find the global maximum
            max_idx = np.argmax(a_lats)
            if max_idx != 0:
                # check if there is a local maximum before max_idx
                d_a_lats = np.diff(a_lats[0:max_idx])
                local_max_idx = np.argwhere(d_a_lats < 0)
                if local_max_idx.size != 0:
                    max_idx = local_max_idx[0]

            # set everything above max_idx to nan
            self.lookup_table[max_idx + 1:, vel_idx] = None

    def plot_lookup(self):
        # Plot the lookup table as a surface3d with velocitz and steering angle on x and y axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r'$v_x$ [m/s]')
        ax.set_ylabel(r'$\delta$ [rad]')
        # add some space between y label and y axis
        ax.yaxis.labelpad = 10
        ax.set_zlabel(r'$a_c$ [m/s$^2$]')
        ax.set_xlim(START_VEL, END_VEL)
        ax.set_ylim(START_STEER, END_STEER)
        ax.set_zlim(0, 10)

        X, Y = np.meshgrid(self.lookup_table[0, 1:], self.lookup_table[1:, 0])
        Z = self.lookup_table[1:, 1:]
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.6, zorder=0)

        azimuth = -160
        elevation = 15
        ax.view_init(elevation, azimuth)
        plt.tight_layout()

        # add an example lookup pair
        steer = 0.1
        vel = 3.0
        # find corresponding indices
        steer_idx = np.where(np.abs(self.lookup_table[1:, 0] - steer) < 0.01)[0]
        vel_idx = np.where(np.abs(self.lookup_table[0, 1:] - vel) < 0.1)[0]
        if len(steer_idx) == 0 or len(vel_idx) == 0:
            steer_idx = 100
            vel_idx = 100
        if len(steer_idx) > 1:
            steer_idx = steer_idx[0]
        if len(vel_idx) > 1:
            vel_idx = vel_idx[0]

        a_lat = self.lookup_table[steer_idx + 1, vel_idx + 1]
        ax.scatter(vel, steer, a_lat, c='r', marker='o', s=20, zorder=10)
        ax.plot([vel, vel], [steer, steer], [0, a_lat], c='r', linestyle='--', zorder=10)
        ax.plot([vel, vel], [START_STEER, steer], [0, 0], c='r', linestyle='--', zorder=10)
        ax.plot([START_VEL, vel], [steer, steer], [0, 0], c='r', linestyle='--', zorder=10)
        plt.show()

    def save_lookup(self):
        """Write into steering_lookup/cfg with the name LookupSteerAngle expects."""
        script_dir = os.path.dirname(os.path.realpath(__file__))
        cfg_dir = os.path.join(script_dir, "..", "steering_lookup", "cfg")
        archive_dir = os.path.join(cfg_dir, "archive")
        os.makedirs(archive_dir, exist_ok=True)

        stamp = datetime.now().strftime("%m%d")
        archived = os.path.join(
            archive_dir, f"{self.table_name}_{stamp}_pacejka_lookup_table.csv")
        np.savetxt(archived, self.lookup_table, delimiter=",")
        print(f"SAVED LOOKUP TABLE TO: {archived}")

        if self.update_latest:
            target = os.path.join(
                cfg_dir, f"{self.table_name}_pacejka_lookup_table.csv")
            np.savetxt(target, self.lookup_table, delimiter=",")
            print(f"UPDATED: {target}\n  -> set LU_table: {self.table_name}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate a MAP steering lookup table from a fitted model.")
    ap.add_argument("model_name",
                    help="id_analyser model, e.g. UNICORN2-0726_pacejka")
    ap.add_argument("table_name",
                    help="output table prefix, e.g. UNICORN2-0726-urethane")
    ap.add_argument("--mu", type=float, default=None,
                    help="floor friction override; omit to use the model's mu")
    ap.add_argument("--no-update-latest", action="store_true")
    a = ap.parse_args()
    LookupGenerator(a.model_name, a.table_name, a.mu,
                    not a.no_update_latest).run_generator()

if __name__ == "__main__":
    main()
