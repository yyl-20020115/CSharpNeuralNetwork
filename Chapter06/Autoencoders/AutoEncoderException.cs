﻿using System;

namespace Autoencoders;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>   Exception for signalling automatic encoder errors. </summary>
///
/// <seealso cref="T:System.Exception"/>
////////////////////////////////////////////////////////////////////////////////////////////////////

public class AutoEncoderException : Exception
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Initializes a new instance of the Autoencoders.AutoEncoderException class.
    /// </summary>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public AutoEncoderException()
    {
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Initializes a new instance of the Autoencoders.AutoEncoderException class. </summary>
    ///
    /// <param name="message">  The message. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public AutoEncoderException(string message)
        : base(message)
    {
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Initializes a new instance of the Autoencoders.AutoEncoderException class.
    /// </summary>
    ///
    /// <param name="message">  The message. </param>
    /// <param name="inner">    The inner. </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public AutoEncoderException(string message, Exception inner)
        : base(message, inner)
    {
    }
}
