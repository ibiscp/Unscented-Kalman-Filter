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
    std_a_ = 2;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1;

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
    previous_timestamp_ = 0;

    //set state dimension
    n_x_ = 5;

    //set augmented state dimension
    n_aug_ = 7;

    //set radar measurement dimension
    n_zrad_ = 3;

    //set laser measurement dimension
    n_zlas_ = 2;

    //define sigma point spreading parameter
    lambda_ = 3 - n_aug_;

    //create matrix with predicted sigma points as columns
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    //create vector for weights
    weights_ = VectorXd(2*n_aug_+1);

    //current NIS for radar
    NIS_radar_ = 0.0;

    //current NIS for laser
    NIS_laser_ = 0.0;
}

UKF::~UKF() {}

/**
 * Initialize variables
 */
void UKF::Initialize(MeasurementPackage meas_package) {

    // initialize state covariance matrix P
    P_ <<   1, 0,  0,  0,  0,
    0,  1, 0,  0,  0,
    0,  0,  1, 0,  0,
    0,  0,  0,  1, 0,
    0,  0,  0,  0,  1;

    //set weights
    double weight_0 = lambda_/(lambda_+n_aug_);
    weights_(0) = weight_0;
    for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
        double weight = 0.5/(n_aug_+lambda_);
        weights_(i) = weight;
    }

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
 * Augment Sigma Points
 */
void UKF::AugmentSigmaPoints(MatrixXd &Xsig_aug) {
    //create augmented mean vector
    static VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    static MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

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
}

/**
 * Predict Sigma Points
 */
void UKF::PredictSigmaPoints(MatrixXd &Xsig_aug, double delta_t, MatrixXd &Xsig_pred) {

    //predict sigma points
    for (int i = 0; i< 2*n_aug_+1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        } else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        //write predicted sigma point into right column
        Xsig_pred(0,i) = px_p;
        Xsig_pred(1,i) = py_p;
        Xsig_pred(2,i) = v_p;
        Xsig_pred(3,i) = yaw_p;
        Xsig_pred(4,i) = yawd_p;
    }
    return;
}

/**
 * Predict Mean and Covariance
 */
void UKF::PredictMeanAndCovariance(MatrixXd &Xsig_pred, VectorXd &x_pred, MatrixXd &P_pred) {
    //predicted state mean
    x_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_pred = x_pred + weights_(i) * Xsig_pred.col(i);
    }

    //predicted state covariance matrix
    P_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred.col(i) - x_;
        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P_pred = P_pred + weights_(i) * x_diff * x_diff.transpose() ;
    }

    return;
}

/**
 * The latest measurement data of either radar or laser.
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
        previous_timestamp_ = meas_package.timestamp_;

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
    double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
    previous_timestamp_ = meas_package.timestamp_;

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

        //mean predicted measurement
        VectorXd z_pred = VectorXd::Zero(n_zrad_);
        //measurement covariance matrix S
        MatrixXd S = MatrixXd::Zero(n_zrad_,n_zrad_);
        // cross-correlation matrix Tc
        MatrixXd Tc = MatrixXd::Zero(n_x_, n_zrad_);
        // get predictions for x,S and Tc in RADAR space
        PredictRadarMeasurement(z_pred, S, Tc);
        // update the state using the RADAR measurement
        UpdateRadar(meas_package, z_pred, Tc, S);
        // update the time
        previous_timestamp_ = meas_package.timestamp_;
    } else if (use_laser_) {
        // Laser updates

        //mean predicted measurement
        VectorXd z_pred = VectorXd::Zero(n_zlas_);
        //measurement covariance matrix S
        MatrixXd S = MatrixXd::Zero(n_zlas_,n_zlas_);
        // cross-correlation matrix Tc
        MatrixXd Tc = MatrixXd::Zero(n_x_, n_zlas_);
        // get predictions for x,S and Tc in Lidar space
        PredictLidarMeasurement(z_pred, S, Tc);
        // update the state using the LIDAR measurement
        UpdateLidar(meas_package, z_pred, Tc, S);
        // update the time
        previous_timestamp_ = meas_package.timestamp_;
    }

    // print the output
    //cout << "x_ = " << x_ << endl;
    //cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

    /// Augment sigma points
    static MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_ + 1);
    AugmentSigmaPoints(Xsig_aug);

    /// Predict sigma points
    static MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
    PredictSigmaPoints(Xsig_aug, delta_t, Xsig_pred);

    /// Predict mean and covariance
    static VectorXd x_pred = VectorXd(n_x_);
    static MatrixXd P_pred = MatrixXd(n_x_, n_x_);
    PredictMeanAndCovariance(Xsig_pred, x_pred, P_pred);

    x_ = x_pred;
    P_ = P_pred;
    Xsig_pred_ = Xsig_pred;
}

void UKF::PredictRadarMeasurement(VectorXd &z_pred, MatrixXd &S, MatrixXd &Tc) {
    //create matrix for sigma points in measurement space
    static MatrixXd Zsig = MatrixXd(n_zrad_, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);
        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // Avoid division by zero
        if(fabs(p_x) <= 0.0001) {
            p_x = 0.0001;
        }
        if(fabs(p_y) <= 0.0001) {
            p_y = 0.0001;
        }

        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
        Zsig(1,i) = atan2(p_y,p_x);                                 //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    //mean predicted measurement
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();

        //state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    static MatrixXd R = MatrixXd(n_zrad_,n_zrad_);
    R << pow(std_radr_,2), 0, 0,
    0, pow(std_radphi_,2), 0,
    0, 0, pow(std_radrd_,2);
    S = S + R;
}

void UKF::PredictLidarMeasurement(VectorXd &z_pred, MatrixXd &S, MatrixXd &Tc) {

    //create matrix for sigma points in measurement space
    static MatrixXd Zsig = MatrixXd(n_zlas_, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        // measurement model
        Zsig(0,i) = Xsig_pred_(0,i);          //px
        Zsig(1,i) = Xsig_pred_(1,i);          //py
    }

    //mean predicted measurement
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        S = S + weights_(i) * z_diff * z_diff.transpose();

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    static MatrixXd R = MatrixXd(n_zlas_,n_zlas_);
    R <<    pow(std_laspx_,2), 0,
    0, pow(std_laspy_,2);
    S = S + R;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package, VectorXd &z_pred, MatrixXd &Tc, MatrixXd &S) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */

    //mean predicted measurement
    static VectorXd z = VectorXd::Zero(n_zlas_);
    z << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1);

    //Kalman gain K
    static MatrixXd K = MatrixXd::Zero(n_x_,n_zlas_);
    K = Tc * S.inverse();

    //residual
    static VectorXd z_diff = VectorXd::Zero(n_zlas_);
    z_diff = z - z_pred;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package, VectorXd &z_pred, MatrixXd &Tc, MatrixXd &S) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */

    //mean predicted measurement  rho, phi, rho_dot
    static VectorXd z = VectorXd::Zero(n_zrad_);
    z << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1),meas_package.raw_measurements_(2);

    //Kalman gain K
    static MatrixXd K = MatrixXd::Zero(n_x_,n_zrad_);
    K = Tc * S.inverse();

    //residual
    static VectorXd z_diff = VectorXd::Zero(n_zrad_);
    z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
