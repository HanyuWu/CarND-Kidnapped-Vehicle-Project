/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include "helper_functions.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using std::normal_distribution;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  std::default_random_engine gen;
  num_particles = 200; // Set the number of particles to 1000 (not default);
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; i++) {
    Particle particle_;
    particle_.id = i;
    particle_.x = dist_x(gen);
    particle_.y = dist_y(gen);
    particle_.theta = dist_theta(gen);
    particle_.weight = 1;
    particles.push_back(particle_);
  }
  is_initialized = true;
  weights = {};
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  double *x_;
  double *y_;
  double *theta_;
  for (int i = 0; i < num_particles; i++) {
    x_ = &particles[i].x;
    y_ = &particles[i].y;
    theta_ = &particles[i].theta;

    // Move particle
    *x_ = *x_ + velocity / yaw_rate *
                    (sin(*theta_ + delta_t * yaw_rate) - sin(*theta_));
    *y_ = *y_ + velocity / yaw_rate *
                    (cos(*theta_) - cos(*theta_ + delta_t * yaw_rate));
    *theta_ = *theta_ + yaw_rate * delta_t;

    // Add Gaussian noise
    *x_ += dist_x(gen);
    *y_ += dist_y(gen);
    *theta_ += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

  for (unsigned int i = 0; i < observations.size(); i++) {
    double l2n_min = std::numeric_limits<double>::max();
    double l2n;
    for (unsigned int j = 0; j < predicted.size(); j++) {
      l2n = dist(observations[i].x, observations[i].y, predicted[j].x,
                 predicted[j].y);
      if (l2n < l2n_min) {
        l2n_min = l2n;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no
   * scaling). The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  for (int i = 0; i < num_particles; i++) {
    vector<LandmarkObs> predicted;
    vector<LandmarkObs> fixed_observation;
    particles[i].weight = 1.0;
    weights = {};

    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float obs_x = map_landmarks.landmark_list[j].x_f;
      float obs_y = map_landmarks.landmark_list[j].y_f;
      int id_ = map_landmarks.landmark_list[j].id_i;
      if (fabs(obs_x - particles[i].x) <= sensor_range &&
          fabs(obs_y - particles[i].y) <= sensor_range) {
        LandmarkObs point{id_, obs_x, obs_y};
        predicted.push_back(point);
      }
    }
    for (unsigned int j = 0; j < observations.size(); j++) {
      double x_m = particles[i].x +
                   (cos(particles[i].theta) * observations[j].x) -
                   (sin(particles[i].theta) * observations[j].y);
      double y_m = particles[i].y +
                   (sin(particles[i].theta) * observations[j].x) +
                   (cos(particles[i].theta) * observations[j].y);
      LandmarkObs fix_point{observations[j].id, x_m, y_m};
      fixed_observation.push_back(fix_point);
    }

    dataAssociation(predicted, fixed_observation);

    for (unsigned int j = 0; j < fixed_observation.size(); j++) {
      int index = fixed_observation[j].id;
      for (unsigned int k = 0; k < predicted.size(); k++) {
        if (index == predicted[k].id) {
          double mu_x = predicted[k].x;
          double mu_y = predicted[k].y;
          particles[i].weight *=
              multiv_prob(sig_x, sig_y, fixed_observation[j].x,
                          fixed_observation[j].y, mu_x, mu_y);
        }
      }
    }
    weights.push_back(particles[i].weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;
  std::default_random_engine gen;
  std::uniform_int_distribution<int> uniintdist(0, num_particles - 1);
  auto index = uniintdist(gen);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  std::uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}