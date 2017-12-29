#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

const double Lf = 2.67;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

Eigen::VectorXd polyfit(std::vector<double> xvals, std::vector<double> yvals,
                        int order)
{
  Eigen::VectorXd ptsx(xvals.size());
  Eigen::VectorXd ptsy(yvals.size());
  assert(xvals.size() == yvals.size() && "polyfit xval yval not same size");
  for (int i = 0; i < ptsx.size(); i++) {
    ptsx(i) = xvals[i];
    ptsy(i) = yvals[i];
  }

  return polyfit(ptsx, ptsy, order);
}

double evalPolynomialGradient(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 1; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i-1);
  }
  return result;
}

void mapToCarFrame(vector<double> car_state,
                              vector<double> &ptsx, vector<double> &ptsy)
{
  for (int i = 0; i < ptsx.size(); i++) {
    double x_diff, y_diff, car_theta;
    car_theta = car_state[2];
    x_diff = ptsx[i] - car_state[0];
    y_diff = ptsy[i] - car_state[1];
    ptsx[i] = cos(car_theta) * x_diff + sin(car_theta) * y_diff;
    ptsy[i] = -sin(car_theta) * x_diff + cos(car_theta) * y_diff;
  }
}


int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    // cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          psi = atan2f(sin(psi), cos(psi)); // unwrap 2pi
          double v = j[1]["speed"];

          // find the appropriate fit order
          int polyfit_order = 3;
          polyfit_order = ptsx.size() < polyfit_order ?
                                              ptsx.size()-1 : polyfit_order;

          // transform to car frame before fitting a polynomial.
          mapToCarFrame({px,py,psi}, ptsx, ptsy);

          px = py = psi = 0;

          auto coeffs = polyfit(ptsx, ptsy, polyfit_order);

          // calculate the cross track error
          double y_ref = polyeval(coeffs, px);
          double cte = py - y_ref;
          // calculate the orientation error
          double desired_heading = atanf(evalPolynomialGradient(coeffs, px));
          double epsi = psi - desired_heading;

          double latency = 0.1;
          double delta = j[1]["steering_angle"];
          delta *= -1;
          double a =  j[1]["throttle"];

          //to convert miles per hour to meter per second
          v *= 0.44704;
          psi = delta; // in coordinate now, so use steering angle to predict x and y
          px = px + v*cos(psi)*latency;
          py = py + v*sin(psi)*latency;
          cte= cte + v*sin(epsi)*latency;
          epsi = epsi + v*delta*latency/Lf;
          psi = psi + v*delta*latency/Lf;
          v = v + a*latency;

          // get the current state
          Eigen::VectorXd state(6);
          state << px, py, psi, v, cte, epsi;

          /*
          * Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          std::vector<double> x_vals = {state[0]};
          std::vector<double> y_vals = {state[1]};
          std::vector<double> psi_vals = {state[2]};
          std::vector<double> v_vals = {state[3]};
          std::vector<double> cte_vals = {state[4]};
          std::vector<double> epsi_vals = {state[5]};
          std::vector<double> delta_vals = {};
          std::vector<double> a_vals = {};

          auto vars = mpc.Solve(state, coeffs);

          const int N = 10;
          const int x_start = 0;
          const int y_start = x_start+N;
          const int psi_start = y_start+N;
          const int v_start = psi_start+N;
          const int cte_start = v_start+N;
          const int epsi_start = cte_start+N;
          const int delta_start = epsi_start + N;
          const int a_start = delta_start + N - 1;

          x_vals.push_back(vars[x_start]);
          y_vals.push_back(vars[y_start]);
          psi_vals.push_back(vars[psi_start]);
          v_vals.push_back(vars[v_start]);
          cte_vals.push_back(vars[cte_start]);
          epsi_vals.push_back(vars[epsi_start]);

          delta_vals.push_back(vars[delta_start]);
          a_vals.push_back(vars[a_start]);

          state << vars[x_start], vars[y_start], vars[psi_start], vars[v_start],
                        vars[cte_start], vars[epsi_start];

          double steer_value = vars[delta_start];
          double throttle_value = vars[a_start];

          steer_value /= -deg2rad(25);

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          for (int i = 0; i < N; i++) {
            mpc_x_vals.push_back(vars[x_start+i]);
            mpc_y_vals.push_back(vars[y_start+i]);
          }
          // mpc_x_vals = ptsx;
          // for (int i = 0; i < ptsx.size(); i++) {
          //   mpc_y_vals.push_back(polyeval(coeffs, ptsx[i]));
          // }

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
          next_x_vals = ptsx;
          next_y_vals = ptsy;

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
