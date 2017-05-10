#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 5;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
    TODO:

    Complete the initialization. See ukf.h for other member properties.

    Hint: one or more values initialized above might be wildly off...
    */

    // variable to indicate if the process is initialized
    is_initialized_ = false;

    // variable for previous timestamp
    time_us_ = 0;

    //set state dimension
    int n_x_ = 5;

    //set augmented dimension
    int n_aug_ = 7;

    //define spreading parameter
    double lambda_ = 3 - n_aug_;

    // predicted sigma points matrix
    Xsig_pred_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create sigma point matrix
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package
 * Initialize variables
 */
void UKF::Initialize(MeasurementPackage meas_package) {
    // initialize state covariance matrix P
    P_ <<   1, 0,  0,  0,  0,
    0,  1, 0,  0,  0,
    0,  0,  1, 0,  0,
    0,  0,  0,  1, 0,
    0,  0,  0,  0,  1;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Convert radar from polar to cartesian coordinates and initialize state.
        float ro = meas_package.raw_measurements_(0);
        float phi = meas_package.raw_measurements_(1);
        float ro_dot = meas_package.raw_measurements_(2);

        float px = ro * cos(phi);
        float py = ro * sin(phi);
        float vx = ro_dot * cos(phi);
        float vy = ro_dot * sin(phi);

        x_(0) = px;
        x_(1) = py;
        x_(2) = sqrt(vx*vx + vy*vy);
        x_(3) = 0;
        x_(4) = 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        // Initialize state.
        x_(0) = meas_package.raw_measurements_(0);
        x_(1) = meas_package.raw_measurements_(1);
        x_(2) = 0;
        x_(3) = 0;
        x_(4) = 0;
    }
    return;
}

/**
 * @param {MeasurementPackage} meas_package
 * Calculates Augment Sigma Points
 */
void UKF::AugmentSigmaPoints(MatrixXd &Xsig_aug) {
    //create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5,5) = P_;
    P_aug(5,5) = std_a_ * std_a_;
    P_aug(6,6) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i+1)     = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }
    return;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    /*****************************************************************************
    *  Initialization
    ****************************************************************************/
    if (!is_initialized_) {
        /**
          * Initialize the state x_ with the first measurement.
          * Create the covariance matrix.
          * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */

        Initialize(meas_package);

        // previous timestamp
        time_us_ = meas_package.timestamp_;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
       * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
       * Update the process noise covariance matrix.
       * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    //compute the time elapsed between the current and previous measurements
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
        // Radar updates
        UpdateRadar(meas_package);
    } else if (use_laser_) {
        // Laser updates
        UpdateLidar(meas_package);
    }

    // print the output
    cout << "x_ = " << x_ << endl;
    cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    /// Augment sigma points
    AugmentSigmaPoints(Xsig_aug)

    /// Predict sigma points
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */
}
