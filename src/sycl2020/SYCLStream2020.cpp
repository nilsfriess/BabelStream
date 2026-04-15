
// Copyright (c) 2015-23 Tom Deakin, Simon McIntosh-Smith, and Tom Lin
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include "SYCLStream2020.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#define ALIGNMENT (1024 * 1024 * 2)

// Cache list of devices
bool cached = false;
std::vector<sycl::device> devices;
void getDeviceList(void);

template <class T>
SYCLStream<T>::SYCLStream(BenchId bs, const intptr_t array_size, const int device_index,
			  T initA, T initB, T initC)
  : array_size(array_size)
{
  if (!cached)
    getDeviceList();

  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");

  sycl::device dev = devices[device_index];

  // Print out device information
  std::cout << "Using SYCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  // Check device can support FP64 if needed
  if (sizeof(T) == sizeof(double))
  {
    if (!dev.has(sycl::aspect::fp64))
    {
      throw std::runtime_error("Device does not support double precision, please use --float");
    }
  }

  queue = std::make_unique<sycl::queue>(dev, sycl::async_handler{[&](sycl::exception_list l)
  {
    bool error = false;
    for(auto e: l)
    {
      try
      {
        std::rethrow_exception(e);
      }
      catch (sycl::exception e)
      {
        std::cout << e.what();
        error = true;
      }
    }
    if(error)
    {
      throw std::runtime_error("SYCL errors detected");
    }
  }});

  // Allocate memory
#if defined(ACCESSOR)
  d_a = std::make_unique<sycl::buffer<T>>(sycl::range<1>{array_size});
  d_b = std::make_unique<sycl::buffer<T>>(sycl::range<1>{array_size});
  d_c = std::make_unique<sycl::buffer<T>>(sycl::range<1>{array_size});
  d_sum = std::make_unique<sycl::buffer<T>>(sycl::range<1>{1});

#elif defined(USM)
#if defined(PAGEFAULT)
  a = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  b = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  c = (T*)aligned_alloc(ALIGNMENT, array_size * sizeof(T));
  sum = (T*)aligned_alloc(ALIGNMENT, ALIGNMENT);
#else
  a = sycl::malloc_shared<T>(array_size, *queue);
  b = sycl::malloc_shared<T>(array_size, *queue);
  c = sycl::malloc_shared<T>(array_size, *queue);
  sum = sycl::malloc_shared<T>(1, *queue);
#endif

#else
  #error unimplemented
#endif
  
  // No longer need list of devices
  devices.clear();
  cached = true;

  init_arrays(initA, initB, initC);
}

template<class T>
SYCLStream<T>::~SYCLStream() {
#if defined(ACCESSOR)
  // Accessor mode owns memory via buffers.
#elif defined(USM)
#if defined(PAGEFAULT)
  free(a);
  free(b);
  free(c);
  free(sum);
#else
  sycl::free(a, *queue);
  sycl::free(b, *queue);
  sycl::free(c, *queue);
  sycl::free(sum, *queue);
#endif
#else
  #error unimplemented
#endif

  devices.clear();
}

template <class T>
void SYCLStream<T>::copy()
{
  queue->submit([&](sycl::handler &cgh)
  {
#if defined(ACCESSOR)
  sycl::accessor a {*d_a, cgh, sycl::read_only};
  sycl::accessor c {*d_c, cgh, sycl::write_only};
#else
    T *a_ptr = a;
    T *c_ptr = c;
#endif
    cgh.parallel_for(sycl::range<1>{array_size},
#if defined(ACCESSOR)
      [a = a, c = c](sycl::id<1> idx)
#else
      [=](sycl::id<1> idx)
#endif
    {
#if defined(ACCESSOR)
      c[idx] = a[idx];
#else
      c_ptr[idx] = a_ptr[idx];
#endif
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::mul()
{
  const T scalar = startScalar;
  queue->submit([&](sycl::handler &cgh)
  {
#if defined(ACCESSOR)
  sycl::accessor b {*d_b, cgh, sycl::write_only};
  sycl::accessor c {*d_c, cgh, sycl::read_only};
#else
    T *b_ptr = b;
    T *c_ptr = c;
#endif
    cgh.parallel_for(sycl::range<1>{array_size},
#if defined(ACCESSOR)
      [=, b = b, c = c](sycl::id<1> idx)
#else
      [=](sycl::id<1> idx)
#endif
    {
#if defined(ACCESSOR)
      b[idx] = scalar * c[idx];
#else
      b_ptr[idx] = scalar * c_ptr[idx];
#endif
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::add()
{
  queue->submit([&](sycl::handler &cgh)
  {
#if defined(ACCESSOR)
  sycl::accessor a {*d_a, cgh, sycl::read_only};
  sycl::accessor b {*d_b, cgh, sycl::read_only};
  sycl::accessor c {*d_c, cgh, sycl::write_only};
#else
    T *a_ptr = a;
    T *b_ptr = b;
    T *c_ptr = c;
#endif
    cgh.parallel_for(sycl::range<1>{array_size},
#if defined(ACCESSOR)
      [a = a, b = b, c = c](sycl::id<1> idx)
#else
      [=](sycl::id<1> idx)
#endif
    {
#if defined(ACCESSOR)
      c[idx] = a[idx] + b[idx];
#else
      c_ptr[idx] = a_ptr[idx] + b_ptr[idx];
#endif
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::triad()
{
  const T scalar = startScalar;
  queue->submit([&](sycl::handler &cgh)
  {
#if defined(ACCESSOR)
  sycl::accessor a {*d_a, cgh, sycl::write_only};
  sycl::accessor b {*d_b, cgh, sycl::read_only};
  sycl::accessor c {*d_c, cgh, sycl::read_only};
#else
    T *a_ptr = a;
    T *b_ptr = b;
    T *c_ptr = c;
#endif
    cgh.parallel_for(sycl::range<1>{array_size},
#if defined(ACCESSOR)
      [=, a = a, b = b, c = c](sycl::id<1> idx)
#else
      [=](sycl::id<1> idx)
#endif
    {
#if defined(ACCESSOR)
      a[idx] = b[idx] + scalar * c[idx];
#else
      a_ptr[idx] = b_ptr[idx] + scalar * c_ptr[idx];
#endif
    });
  });
  queue->wait();
}

template <class T>
void SYCLStream<T>::nstream()
{
  const T scalar = startScalar;
  queue->submit([&](sycl::handler &cgh)
  {
#if defined(ACCESSOR)
  sycl::accessor a {*d_a, cgh};
  sycl::accessor b {*d_b, cgh, sycl::read_only};
  sycl::accessor c {*d_c, cgh, sycl::read_only};
#else
    T *a_ptr = a;
    T *b_ptr = b;
    T *c_ptr = c;
#endif
    cgh.parallel_for(sycl::range<1>{array_size},
#if defined(ACCESSOR)
      [=, a = a, b = b, c = c](sycl::id<1> idx)
#else
      [=](sycl::id<1> idx)
#endif
    {
#if defined(ACCESSOR)
      a[idx] += b[idx] + scalar * c[idx];
#else
      a_ptr[idx] += b_ptr[idx] + scalar * c_ptr[idx];
#endif
    });
  });
  queue->wait();
}

template <class T>
T SYCLStream<T>::dot()
{
#if defined(ACCESSOR)
  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor a {*d_a, cgh, sycl::read_only};
    sycl::accessor b {*d_b, cgh, sycl::read_only};
    cgh.parallel_for(sycl::range<1>{array_size},
      // Reduction object, to perform summation - initialises the result to zero
      // AdaptiveCpp doesn't sypport the initialize_to_identity property yet
#if defined(__HIPSYCL__) || defined(__OPENSYCL__) || defined(__ADAPTIVECPP__)
  sycl::reduction(*d_sum, cgh, sycl::plus<T>()),
#else
  sycl::reduction(*d_sum, cgh, sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
#endif
      [a = a, b = b](sycl::id<1> idx, auto &sum)
      {
        sum += a[idx] * b[idx];
      });
  });
  queue->wait();
  sycl::host_accessor h_sum {*d_sum, sycl::read_only};
  return h_sum[0];
#else
  *sum = static_cast<T>(0);
  queue->submit([&](sycl::handler &cgh)
  {
    T *a_ptr = a;
    T *b_ptr = b;
    cgh.parallel_for(sycl::range<1>{array_size},
      // Reduction object, to perform summation - initialises the result to zero
      // AdaptiveCpp doesn't support the initialize_to_identity property yet
#if defined(__HIPSYCL__) || defined(__OPENSYCL__) || defined(__ADAPTIVECPP__)
      sycl::reduction(sum, sycl::plus<T>()),
#else
      sycl::reduction(sum, sycl::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
#endif
      [=](sycl::id<1> idx, auto &acc)
      {
        acc += a_ptr[idx] * b_ptr[idx];
      });
  });
  queue->wait();
  return *sum;
#endif
}

template <class T>
void SYCLStream<T>::init_arrays(T initA, T initB, T initC)
{
#if defined(USM) && defined(PAGEFAULT)
  for (size_t i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
#else
  queue->submit([&](sycl::handler &cgh)
  {
#if defined(ACCESSOR)
  sycl::accessor a {*d_a, cgh, sycl::write_only, sycl::no_init};
  sycl::accessor b {*d_b, cgh, sycl::write_only, sycl::no_init};
  sycl::accessor c {*d_c, cgh, sycl::write_only, sycl::no_init};
#else
    T *a_ptr = a;
    T *b_ptr = b;
    T *c_ptr = c;
#endif
    cgh.parallel_for(sycl::range<1>{array_size},
#if defined(ACCESSOR)
      [=, a = a, b = b, c = c](sycl::id<1> idx)
#else
      [=](sycl::id<1> idx)
#endif
    
    {
#if defined(ACCESSOR)
      a[idx] = initA;
      b[idx] = initB;
      c[idx] = initC;
#else
      a_ptr[idx] = initA;
      b_ptr[idx] = initB;
      c_ptr[idx] = initC;
#endif
    });
  });
  queue->wait();
#endif
}

template <class T>
void SYCLStream<T>::get_arrays(T const*& h_a, T const*& h_b, T const*& h_c)
{
#if defined(ACCESSOR)
  sycl::host_accessor a {*d_a, sycl::read_only};
  sycl::host_accessor b {*d_b, sycl::read_only};
  sycl::host_accessor c {*d_c, sycl::read_only};
  host_a.assign(a.begin(), a.end());
  host_b.assign(b.begin(), b.end());
  host_c.assign(c.begin(), c.end());
  h_a = host_a.data();
  h_b = host_b.data();
  h_c = host_c.data();
#else
  h_a = a;
  h_b = b;
  h_c = c;
#endif
}

void getDeviceList(void)
{
  // Ask SYCL runtime for all devices in system
  devices = sycl::device::get_devices();
  cached = true;
}

void listDevices(void)
{
  getDeviceList();

  // Print device names
  if (devices.size() == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < devices.size(); i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}

std::string getDeviceName(const int device)
{
  if (!cached)
    getDeviceList();

  std::string name;

  if (device < devices.size())
  {
    name = devices[device].get_info<sycl::info::device::name>();
  }
  else
  {
    throw std::runtime_error("Error asking for name for non-existant device");
  }

  return name;
}

std::string getDeviceDriver(const int device)
{
  if (!cached)
    getDeviceList();

  std::string driver;

  if (device < devices.size())
  {
    driver = devices[device].get_info<sycl::info::device::driver_version>();
  }
  else
  {
    throw std::runtime_error("Error asking for driver for non-existant device");
  }

  return driver;
}

template class SYCLStream<float>;
template class SYCLStream<double>;
