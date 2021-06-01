/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/
#include <iostream>
#include <vector>
#include <thread>

struct Foo
{
};

void
printThreadIDs (const Foo & f)
{
    std::cout << "Thread ID: " << std::this_thread::get_id () << std::endl;
}

int
main ()
{

    std::vector < std::thread > threads;

    for (int i = 0; i < 4; i++)
    {
      threads.push_back (std::thread (printThreadIDs, Foo ()));
      // Instead of copying, move t into the vector
    }

  // Now wait for the threads to finish,
  // We need to wait otherwise main thread might reach an end before the multiple threads finish their work
    for (auto & t:threads)
    {
        t.join ();
    }

}
