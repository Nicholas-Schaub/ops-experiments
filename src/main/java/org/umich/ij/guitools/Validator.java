package org.umich.ij.guitools;

public abstract interface Validator<T>
{
  public abstract boolean validate(String paramString);
  
  public abstract String getErrorText();
  
  public abstract T getValue(String paramString);
}