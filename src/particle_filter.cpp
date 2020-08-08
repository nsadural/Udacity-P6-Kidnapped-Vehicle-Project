/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;
using std::max_element;
using std::cout;
using std::endl;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine random_gen;

  num_particles = 500;  // Set arbitrary, sufficient number of particles

  // Create normal distributions for initial GPS means and stds
  normal_distribution<double> norm_dist_x(x, std[0]);
  normal_distribution<double> norm_dist_y(y, std[1]);
  normal_distribution<double> norm_dist_theta(theta, std[2]);
  
  // Create and populate set of particles using Particle data structure using initial state estimates
  Particle particle_i;  // holds members of i-th Particle

  for (int i = 0; i < num_particles; ++i) {
    particle_i.id = i;
    particle_i.x = norm_dist_x(random_gen);
    particle_i.y = norm_dist_y(random_gen);
    particle_i.theta = norm_dist_theta(random_gen);
    particle_i.weight = 1.0;
    weights.push_back(1.0);
    particles.push_back(particle_i);
  }

  // Set initialize flag to true
  is_initialized = true;
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

  std::default_random_engine random_gen;

  // Define random Gaussian sensor noise with zero mean and add to each particle
  normal_distribution<double> noise_x(0.0, std_pos[0]);
  normal_distribution<double> noise_y(0.0, std_pos[1]);
  normal_distribution<double> noise_theta(0.0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {

      if (fabs(yaw_rate) > 0.0001) {
        particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
        particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
        particles[i].theta += yaw_rate * delta_t;
      } else {
        particles[i].x += velocity * cos(particles[i].theta) * delta_t;
        particles[i].y += velocity * sin(particles[i].theta) * delta_t;
      }

      particles[i].x += noise_x(random_gen);
      particles[i].y += noise_y(random_gen);
      particles[i].theta += noise_theta(random_gen);
      
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double curr_dist;
  double min_dist;

  for (unsigned int i = 0; i < observations.size(); ++i) {

      min_dist = std::numeric_limits<double>::max();

      for (unsigned int j = 0; j < predicted.size(); ++j) {
          
          curr_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
          
          if (curr_dist < min_dist) {
              observations[i].id = predicted[j].id;
              min_dist = curr_dist;
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
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */    
  double gauss_exp;
  double gauss_norm;

  for (int i = 0; i < num_particles; ++i) {
      
      vector<LandmarkObs> tf_observations;

      for (unsigned int j = 0; j < observations.size(); ++j) {
        LandmarkObs tf_obs;
        tf_obs.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
        tf_obs.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
        tf_obs.id = observations[j].id;
        tf_observations.push_back(tf_obs);
      }

      vector<LandmarkObs> predicted;

      for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
        int pd_id = map_landmarks.landmark_list[k].id_i;
        double pd_x = map_landmarks.landmark_list[k].x_f;
        double pd_y = map_landmarks.landmark_list[k].y_f;

        if (dist(particles[i].x, particles[i].y, pd_x, pd_y) <= sensor_range) {
          LandmarkObs pd_in_range;
          pd_in_range.id = pd_id;
          pd_in_range.x = pd_x;
          pd_in_range.y = pd_y;
          predicted.push_back(pd_in_range);
        }

      }

      dataAssociation(predicted, tf_observations);

      double weight = 1.0;
      particles[i].weight = weight;

      double landmark_x;
      double landmark_y;
      
      gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);

      for (unsigned int j = 0; j < tf_observations.size(); ++j) {
          
        for (unsigned int k = 0; k < predicted.size(); ++k) {
          
          if (tf_observations[j].id == predicted[k].id) {
            
            landmark_x = predicted[k].x;
            landmark_y = predicted[k].y;
            gauss_exp = (pow(tf_observations[j].x - landmark_x, 2.0) / (2.0 * pow(std_landmark[0], 2.0))) + (pow(tf_observations[j].y - landmark_y, 2.0) / (2.0 * pow(std_landmark[1], 2.0)));
            weight = weight * gauss_norm * exp(-gauss_exp);
            break;
          }

        }

      }
      particles[i].weight = weight;
      weights[i] = weight;      
  }

  double weights_sum = 0.0;
  for (int i = 0; i < num_particles; ++i) {
    weights_sum += weights[i];
  }

  for (int i = 0; i < num_particles; ++i) {
    weights[i] = weights[i] / weights_sum;
    particles[i].weight = weights[i];
  }

}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::default_random_engine random_gen;
  discrete_distribution<int> index_dist(weights.begin(), weights.end());
  vector<Particle> new_particles;

  for (int i = 0; i < num_particles; ++i) {
    int index = index_dist(random_gen);
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
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
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}