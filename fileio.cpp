#include "Python.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cctype>
#include <ctime>

using namespace std;

struct Spectrum
{
    char title[256];
    double pepmass;
    int charge;
    double retsec;
    char sequence[256];
    int length;
    int peak;
    double* mz;
    double* intensity;
};

void clear_spectrum(Spectrum* spectrum)
{
    // if (spectrum->sequence != NULL)
    // {
    //     delete[] spectrum->sequence;
    //     spectrum->sequence = NULL;
    // }
    if (spectrum->mz != NULL)
    {
        delete[] spectrum->mz;
        spectrum->mz = NULL;
    }
    if ((spectrum->intensity != NULL))
    {
        delete[] spectrum->intensity;
        spectrum->intensity = NULL;
    }
}

void clear_spectra(vector<Spectrum*>& spectra)
{
    for (auto spectrum : spectra)
    {
        clear_spectrum(spectrum);
        delete spectrum;
        spectrum = NULL;
    }
}

void parse_header(Spectrum& peptide, const char* line)
{
    char key[16], value[256];
    sscanf(line, "%[^=]=%s", &key, &value);
    if (!strcmp(key, "TITLE"))
    {
        strcpy(peptide.title, value);
    }
    else if (!strcmp(key, "PEPMASS"))
    {
        peptide.pepmass = atof(value);
    }
    else if (!strcmp(key, "CHARGE"))
    {
        peptide.charge = atoi(value);//最后那个+不影响
    }
    else if (!strcmp(key, "RTINSECONDS"))
    {
        peptide.retsec = atof(value);
    }
    else if (!strcmp(key, "SEQ"))
    {
        //peptide.sequence = new char[strlen(value)];
        strcpy(peptide.sequence, value);
    }
}

void parse_ms2(vector<double>& mz, vector<double>& intensity, const char* line)
{
    double m, i;
    sscanf(line, "%lf%lf", &m, &i);
    mz.push_back(m);
    intensity.push_back(i);
}

Spectrum* parse_spectrum(ifstream& filehandle, vector<double>& mz, vector<double>& intensity)
{
    Spectrum* spectrum = new Spectrum{};
    char cache[256];

    filehandle.getline(cache, 256);
    if (strcmp(cache, "BEGIN IONS")) throw(-1);//不是才抛出异常   
    while (!isdigit(cache[0]))
    {
        parse_header(*spectrum, cache);
        filehandle.getline(cache, 256);
    }

    while (strcmp(cache, "END IONS"))
    {
        parse_ms2(mz, intensity, cache);
        filehandle.getline(cache, 256);//end后面可能有空行，跳出循环的时候end已经读了
    }

    int i = 0;
    int length = 0;
    while (spectrum->sequence[i])
    {
        if (isalpha(spectrum->sequence[i])) ++length;
        i++;
    }

    spectrum->length = length;

    spectrum->peak = mz.size();
    spectrum->mz = new double[spectrum->peak];
    spectrum->intensity = new double[spectrum->peak];
    memcpy(spectrum->mz, &mz[0], spectrum->peak * sizeof(double));
    memcpy(spectrum->intensity, &intensity[0], spectrum->peak * sizeof(double));

    mz.clear();
    intensity.clear();
    return spectrum;
}

vector<Spectrum*> parse_file(const char* path)
{
    vector<Spectrum*> spectra;
    spectra.reserve(5000);

    ifstream filehandle;
    filehandle.open(path);

    vector<double> mz, intensity;
    mz.reserve(1024);
    intensity.reserve(1024);

    while (!filehandle.eof())
    {
        try
        {
            spectra.push_back(parse_spectrum(filehandle, mz, intensity));
        }
        catch (int)
        {
            cout << "Invalid header" << endl;
            break;
        }

        while (filehandle.peek() == '\n')//blank line, has only a character "\n", with ASCII code 10.
        {
            filehandle.get();
        }
    }
    filehandle.close();
    return spectra;
}
